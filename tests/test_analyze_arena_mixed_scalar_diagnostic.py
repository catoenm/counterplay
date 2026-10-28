"""Independent diagnostic checks using real tiny CPU Torch reports, never models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import analyze_arena_mixed_scalar as audit
from selfplay.arena_mixed_scalar import cuda_policy as backend
from tests.test_arena_mixed_scalar_cuda_policy import fixture, diagnostic_fixture


class DiagnosticAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.cuda_rng=[torch.get_rng_state().clone()]
        self.patches=[patch.object(torch.cuda,'get_rng_state_all',side_effect=lambda:[x.clone() for x in self.cuda_rng]),
            patch.object(torch.cuda,'set_rng_state_all',side_effect=lambda values:self.cuda_rng.__setitem__(slice(None),[x.clone() for x in values]))]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.policy=fixture();self.base=self.policy.save_checkpoint(Path(self.tmp.name)/'base','base',0)
        self.fixture=diagnostic_fixture()
        self.report=self.policy.execution_diagnostic(self.fixture)
        self.assertTrue(self.report['passed'])

    def check(self,report=None,fixture_value=None,base=None):
        with patch.object(backend,'select_labels',return_value=(self.policy.labels,self.policy.label_ids)):
            return audit.check_execution_diagnostic(report or self.report,fixture_value or self.fixture,
                base or self.base,self.policy.config,self.policy.tokenizer)

    def test_actual_saved_base_full_vectors_derivative_and_json_roundtrip(self):
        restored=json.loads(backend.canonical(self.report))
        result=self.check(restored)
        self.assertTrue(result['passed']);self.assertTrue(result['restored_exact_base'])
        self.assertEqual(result['scalar_gate']['records'],10)
        self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(result['report_sha256'],backend.digest(restored))

    def test_rejects_source_geometry_native_tokens_and_restored_state_corruption(self):
        mutations=[
            lambda r:r.__setitem__('fixture_sha256','0'*64),
            lambda r:r['binding'].__setitem__('version',1),
            lambda r:r['groups']['actor_requests'].__setitem__('legacy_physical_batch_size',2),
            lambda r:r['groups']['gate_requests']['records'][0].__setitem__('input_ids_sha256','f'*64),
            lambda r:r['state_after'].__setitem__('model_training',True),
            lambda r:r['state_before']['optimizer'].__setitem__('step',1),
            lambda r:r['scalar_smoke']['rows'][0]['input_ids'].__setitem__(0,999),
            lambda r:r['scalar_smoke']['rows'][0].__setitem__('physical_execution_contract_sha256','e'*64),
        ]
        for i,mutate in enumerate(mutations):
            with self.subTest(case=i):
                bad=deepcopy(self.report);mutate(bad)
                with self.assertRaises(ValueError):self.check(bad)
        badbase=deepcopy(self.base);badbase['rng']={'different':'saved RNG'}
        with self.assertRaisesRegex(ValueError,'fresh base'):self.check(base=badbase)

    def test_rejects_vector_coverage_result_loss_and_derivative_corruption(self):
        mutations=[
            lambda r:r['groups']['actor_requests']['records'].pop(),
            lambda r:r['groups']['gate_requests']['records'][0]['scalar_log_probs'].pop(),
            lambda r:r['groups']['gate_requests']['records'][0].__setitem__('max_error',1.),
            lambda r:r['scalar_smoke']['gate']['records'].pop(),
            lambda r:r['scalar_smoke']['gate']['records'][0].__setitem__('grad_enabled',True),
            lambda r:r['scalar_smoke']['derivative'].__setitem__('loss',999.),
            lambda r:r['scalar_smoke']['derivative'].__setitem__('gradient_norm',0.),
            lambda r:r['scalar_smoke']['rows'][0].__setitem__('action_token_id',999),
        ]
        for i,mutate in enumerate(mutations):
            with self.subTest(case=i):
                bad=deepcopy(self.report);mutate(bad)
                with self.assertRaises(ValueError):self.check(bad)

    def test_legacy_large_geometry_errors_are_descriptive_not_admission_threshold(self):
        report=deepcopy(self.report)
        for name,group in report['groups'].items():
            for row in group['records']:
                values=torch.tensor(row['legacy_log_probs'],dtype=torch.float64)
                values[0]+=.5 if name=='actor_requests' else .1
                row['legacy_log_probs']=torch.log_softmax(values,-1).tolist()
                row['max_error']=max(abs(a-b) for a,b in zip(row['legacy_log_probs'],row['scalar_log_probs']))
        a=report['groups']['actor_requests']['records'][6]['legacy_log_probs']
        b=report['groups']['gate_requests']['records'][0]['legacy_log_probs']
        report['outlier_comparison']['legacy_b8_b2_max_error']=max(abs(x-y) for x,y in zip(a,b))
        self.assertGreater(report['outlier_comparison']['legacy_b8_b2_max_error'],.002)
        self.assertTrue(self.check(report)['passed'])
        report['outlier_comparison']['legacy_condition_selected']=True
        with self.assertRaises(ValueError):self.check(report)

    def test_partial_failed_diagnostic_is_preserved_without_invented_admission(self):
        report=deepcopy(self.report);report['groups']={}
        report.pop('outlier_comparison');report.pop('scalar_smoke')
        report.update(passed=False,error_type='RuntimeError',reason='synthetic failure before comparison')
        result=self.check(report)
        self.assertFalse(result['passed']);self.assertIsNone(result['scalar_gate'])
        report['passed']=True
        with self.assertRaisesRegex(ValueError,'False successful'):self.check(report)

    def test_full_scalar_gate_failure_is_auditable_but_cannot_pass(self):
        report=deepcopy(self.report);smoke=report['scalar_smoke']
        gate=smoke['gate'];record=gate['records'][0]
        record['actual_log_probs'][0]+=.01
        record['max_error']=max(abs(a-b) for a,b in zip(record['actual_log_probs'],smoke['rows'][0]['old_log_probs']))
        record['passed']=False;gate['passed']=False;smoke['passed']=False;report['passed']=False
        result=self.check(report)
        self.assertFalse(result['passed']);self.assertEqual(len(result['scalar_gate']['failures']),1)
        report['passed']=True
        with self.assertRaisesRegex(ValueError,'False successful'):self.check(report)


if __name__=='__main__':unittest.main()
