"""Scalar execution contract with tiny CPU Torch tensors; no pretrained weights."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from selfplay.arena_mixed import cuda_policy as previous
from selfplay.arena_mixed_scalar import cuda_policy as scalar
from tests.test_arena_mixed_cuda_policy import fixture as old_fixture, request, episodes, WEIGHTS


def fixture(role="learner", **config):
    obj = old_fixture(role)
    obj.__class__ = scalar.CudaActionPolicy
    obj.config = scalar.default_config(WEIGHTS, model_index_sha256="a"*64,
        tokenizer_files_sha256="b"*64, **config)
    obj.runtime["physical_execution_contract"] = deepcopy(scalar.PHYSICAL_EXECUTION_CONTRACT)
    return obj


def diagnostic_fixture():
    actor = [request(seed=100+i, extra="request "*(i+1)) for i in range(8)]
    actor[6] = request("chess", seed=106, extra="historical outlier geometry")
    gate = [deepcopy(actor[6]), request("chess", seed=107)]
    smoke = [deepcopy(actor[6]) if env=="chess" else request(env, seed=200+i)
             for i,env in enumerate(WEIGHTS)]
    return {"kind":"mixed_scalar_execution_fixtures_v1",
        "physical_execution_contract":deepcopy(scalar.PHYSICAL_EXECUTION_CONTRACT),
        "outlier_actor_index":6,"outlier_gate_index":0,
        "actor_requests":actor,"gate_requests":gate,"smoke_requests":smoke}


class ScalarCudaTests(unittest.TestCase):
    def setUp(self):
        self.cuda_rng = [torch.get_rng_state().clone()]
        self.patches = [patch.object(torch.cuda,"get_rng_state_all",side_effect=lambda:[x.clone() for x in self.cuda_rng]),
            patch.object(torch.cuda,"set_rng_state_all",side_effect=lambda x:self.cuda_rng.__setitem__(slice(None),[y.clone() for y in x]))]
        for p in self.patches:p.start()
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
    def base(self,obj):
        folder=Path(self.tmp.name)/('base-'+str(id(obj)))
        return folder,obj.save_checkpoint(folder,'base',0)

    def test_exact_configuration_and_fresh_seed(self):
        cfg=scalar.default_config(WEIGHTS)
        self.assertEqual(cfg['seed'],742001);self.assertEqual(cfg['likelihood_tolerance'],.002)
        self.assertEqual(cfg['physical_execution_contract'],scalar.PHYSICAL_EXECUTION_CONTRACT)
        for key,value in [('physical_execution_contract',None),('likelihood_tolerance',.004),
                          ('microbatch_size',1),('inference_batch_size',2)]:
            bad=deepcopy(cfg);bad[key]=value
            with self.assertRaises(ValueError):scalar.validate_config(bad)
        cfg['physical_execution_contract']['physical_batch_size']=2
        with self.assertRaises(ValueError):scalar.validate_config(cfg)
        self.assertEqual(scalar.PHYSICAL_EXECUTION_CONTRACT['physical_batch_size'],1)

    def test_logical_batch_executes_own_length_single_rows_and_full_vectors(self):
        obj=fixture();self.base(obj)
        requests=[request(seed=11,actions=2),request(seed=12,actions=4,extra='longer '*50),request(seed=13,actions=3)]
        rows=obj.choose(requests,obj.binding)
        self.assertEqual(len(obj.model.model.calls),3)
        self.assertEqual(obj.model.lm_head.shapes,[(1,3)]*3)
        for call,row in zip(obj.model.model.calls,rows):
            self.assertEqual(tuple(call['mask'].shape),(1,len(row['input_ids'])))
            self.assertTrue(bool(call['mask'].all()))
            self.assertTrue(torch.equal(call['positions'],torch.arange(len(row['input_ids']))[None,:]))
            single=obj.choose([requests[rows.index(row)]],obj.binding)[0]
            self.assertEqual(row['old_log_probs'],single['old_log_probs'])
            self.assertEqual(len(row['old_log_probs']),len(row['actions']))
            self.assertEqual(row['physical_execution_contract_sha256'],scalar.digest(scalar.PHYSICAL_EXECUTION_CONTRACT))
        reversed_rows=obj.choose(list(reversed(requests)),obj.binding)
        self.assertEqual([x['old_log_probs'] for x in rows],list(reversed([x['old_log_probs'] for x in reversed_rows])))

    def test_cat_keeps_all_per_row_gradients_and_no_cross_row_geometry(self):
        obj=fixture();self.base(obj)
        rows=[obj._request(request(seed=20+i,extra='length '*((i+1)*7))) for i in range(3)]
        obj.model.train();params=list(obj.parameters.values())
        loss=-sum((i+1)*obj.log_probs(rows)[i,i%3] for i in range(3))
        together=torch.autograd.grad(loss,params)
        individual=[torch.zeros_like(p) for p in params]
        for i,row in enumerate(rows):
            one=torch.autograd.grad(-(i+1)*obj.log_probs([row])[0,i%3],params)
            individual=[x+y for x,y in zip(individual,one)]
        for a,b in zip(together,individual):self.assertTrue(torch.allclose(a,b,rtol=1e-5,atol=1e-6))
        self.assertTrue(any(float(g.abs().max())>0 for g in together))
        self.assertTrue(all(c['mask'].shape[0]==1 for c in obj.model.model.calls))

    def test_full_gate_update_checkpoint_and_actor_restore_contract(self):
        obj=fixture();folder,base=self.base(obj);data=episodes(obj);binding=deepcopy(obj.binding)
        raw=[x for e in data for x in e['transitions']]
        gate=obj.full_likelihood_gate(raw,binding)
        self.assertTrue(gate['passed']);self.assertEqual(len(gate['records']),2*len(raw))
        self.assertTrue(all(x['max_error']==0 for x in gate['records']))
        changed=deepcopy(raw[0]);changed.pop('physical_execution_contract_sha256')
        with self.assertRaises(ValueError):obj._validate_row(changed,binding)
        update=obj.train_cohort(data,binding,gate)
        self.assertEqual(update['optimizer_steps'],1);self.assertEqual(len(update['learner_records']),len(raw))
        out=Path(self.tmp.name)/'cycle1';state=obj.save_checkpoint(out,'cycle1',1)
        self.assertEqual(scalar.checkpoint_state(out)['step'],1)
        before=obj._state_identity();replayed=obj.gate_at_checkpoint(raw,folder,base['files'])
        self.assertTrue(replayed['passed']);self.assertEqual(before,obj._state_identity())
        actor=fixture('actor');actor.load_checkpoint(out,state['files'])
        self.assertEqual(actor._optimizer_identity(),obj._optimizer_identity())
        with self.assertRaisesRegex(ValueError,'Actor'):actor.train_cohort(data,binding,gate)

    def test_scalar_smoke_and_temporary_actual_profile_restore_exact_state(self):
        obj=fixture();self.base(obj);before=obj._state_identity()
        smoke=obj.numerical_smoke([request(env,seed=30+i) for i,env in enumerate(WEIGHTS)])
        self.assertTrue(smoke['passed']);self.assertEqual(before,obj._state_identity())
        data=episodes(obj);raw=[x for e in data for x in e['transitions']]
        report=obj.profile_cohort(data,obj.binding,obj.full_likelihood_gate(raw,obj.binding),Path(self.tmp.name)/'profile')
        self.assertTrue(report['passed']);self.assertEqual(report['temporary_optimizer_steps'],1)
        self.assertEqual(report['retained_optimizer_steps'],0);self.assertEqual(before,obj._state_identity())
        self.assertEqual(scalar.checkpoint_state(Path(self.tmp.name)/'profile')['step'],1)

    def test_diagnostic_is_one_descriptive_legacy_comparison_and_scalar_admission(self):
        obj=fixture();self.base(obj);before=obj._state_identity()
        original=previous.CudaActionPolicy.log_probs
        def geometry_sensitive(instance,rows):
            vector=original(instance,rows)
            if len(rows)>1:
                offset=torch.zeros_like(vector);offset[:,0]=.02
                vector=vector+offset
            return vector
        with patch.object(previous.CudaActionPolicy,'log_probs',geometry_sensitive):
            report=obj.execution_diagnostic(diagnostic_fixture())
        self.assertTrue(report['passed']);self.assertEqual(report['optimizer_steps'],0)
        self.assertEqual(before,obj._state_identity());self.assertFalse(report['training_use'])
        self.assertEqual(report['fixture_sha256'],scalar.digest(diagnostic_fixture()))
        self.assertGreater(report['groups']['actor_requests']['records'][0]['max_error'],.002)
        self.assertEqual(report['outlier_comparison']['scalar_group_max_error'],0)
        self.assertFalse(report['outlier_comparison']['legacy_condition_selected'])
        self.assertEqual(len(report['scalar_smoke']['gate']['records']),10)
        with self.assertRaisesRegex(ValueError,'retried'):obj.execution_diagnostic(diagnostic_fixture())

    def test_diagnostic_rejects_repair_or_missing_coverage_and_preserves_failure(self):
        obj=fixture();self.base(obj);bad=diagnostic_fixture();bad['gate_requests'][0]['sample_seed']+=1
        with self.assertRaises(ValueError):obj.execution_diagnostic(bad)
        before=obj._state_identity()
        with patch.object(obj,'numerical_smoke',return_value={'passed':False,'reason':'synthetic gate mismatch'}):
            report=obj.execution_diagnostic(diagnostic_fixture())
        self.assertFalse(report['passed']);self.assertTrue(report['state_restored'])
        self.assertEqual(before,obj._state_identity());self.assertTrue(obj.poisoned)
        self.assertEqual(report['optimizer_steps'],0)
        with self.assertRaises(ValueError):obj.execution_diagnostic(diagnostic_fixture())

    def test_durable_json_and_wrong_contract_rejected(self):
        obj=fixture();self.base(obj);row=obj.choose([request()],obj.binding)[0]
        obj._validate_row(json.loads(scalar.canonical(row)),obj.binding)
        row['physical_execution_contract_sha256']='f'*64
        with self.assertRaises(ValueError):obj._validate_row(row,obj.binding)


if __name__=='__main__':unittest.main()
