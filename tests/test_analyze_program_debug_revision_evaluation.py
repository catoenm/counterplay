"""CPU-only independent-audit fixtures; synthetic draws are never research data."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import tempfile
import unittest

from scripts import analyze_program_debug_revision_evaluation as audit
from selfplay import program_debug_revision_evaluation as runner
from tests.test_analyze_program_debug_revision_rl import SyntheticJournalBackend


class Backend(SyntheticJournalBackend):
    def sample_messages_batch(self,*args,**kwargs):
        result=super().sample_messages_batch(*args,**kwargs)
        for row in result['records']:
            if row['authoring_stage']=='revision' and row['seed']==241000001:
                text=("def act():\n    return 'rest'" if self.identity['step']==0
                      else "def act():\n    v_bad = 1 // 0\n    return 'rest'")
                ids=self.tokenizer.encode(text,add_special_tokens=False)+[248046]
                row.update(completion_ids=ids,output_tokens=len(ids),sampled_token_logprobs=[-.1]*len(ids),
                    text=self.tokenizer.decode(ids,skip_special_tokens=True),
                    raw_text=self.tokenizer.decode(ids,skip_special_tokens=False))
        lengths=[r['output_tokens'] for r in result['records']]
        counts=[sum(n>i for n in lengths) for i in range(max(lengths))]
        for row in result['records']:row['post_eos_cache_forwards']=len(counts)-row['output_tokens']
        result['timing'].update(total_output_tokens=sum(lengths),active_sequences_by_step=counts,batch_steps=len(counts))
        return result

    def mechanics_smoke(self,request):
        seeds=[70000000+self.config['seed']+i for i in range(2)]
        batch=self.sample_messages_batch(request['batch_messages'],seeds,stages=['draft','revision'],maximum=32)
        return {**self.vectors(batch['records']),'kind':'revision_mixed_native_derivative_admission',
            'batch_size':2,'max_new_tokens':32,'authoring_stages':['draft','revision'],'seeds':seeds,'batch':batch,
            'derivative':{'finite_nonzero':True,'gradient_l2_norm':.1,'missing_gradient_names':[]},'fixture_only':True}
    def update_trajectories(self,*args,**kwargs):raise AssertionError('No evaluation optimizer operation')
    full_backward_probe=update_trajectories


def prepared(folder):
    folder=Path(folder);dep=runner.common.load(runner.PARENT/'deployment.json')
    dep.update(maximum_hours=10,reservation_usd=65,estimate_usd=55,
        deadline_unix=int((datetime.now(timezone.utc)+timedelta(hours=10)).timestamp()),
        phase='coding_evaluation',continuation_id='a'*32,combined_funding_usd=240.,intent='d'*32,
        funding_training_intent='b'*32,evaluation_intent='c'*32,coding_evaluation_intent='d'*32,
        funding_training_deployment='/fixture/training',evaluation_deployment='/fixture/evaluation',
        coding_evaluation_deployment='/fixture/coding',predecessor_deployment=str(runner.PARENT),
        predecessor_evidence_sha256={'fixture':'unused'},mixed_controller_sha256='fixture',inherited_controller_sha256='fixture')
    runner.common.atomic_json(folder/'deployment.json',dep)
    run=folder/'run';runner.prepare(run,folder/'deployment.json')
    return run


class EvaluationAuditTests(unittest.TestCase):
    def test_unchanged_state_proof_rejects_hidden_optimizer_or_rng_changes(self):
        source={'step':3,'learning_rate':1e-5,'adapter_tensors':{'q':'x'},'state_sha256':'a','rng':'r','files':{'x':'old'}}
        target={**source,'files':{'x':'new'}}
        proof={'actual_identity':target,'unchanged_adapter_Adam_RNG':True,'new_optimizer_steps':0}
        self.assertTrue(audit.validate_unchanged(proof,source,target))
        for key,value in [('step',4),('rng','other'),('state_sha256','mutated'),('adapter_tensors',{'q':'different'})]:
            bad=deepcopy(target);bad[key]=value
            with self.assertRaisesRegex(ValueError,'mislabeled'):
                audit.validate_unchanged({**proof,'actual_identity':bad},source,bad)

    def test_paired16_retains_invalid_and_runtime_fault_yields_not_missing_draws(self):
        base=[{'generation':f'base-{i}','authorship_valid':False,'runtime_valid':False,'games':0,'banked_point_yield':0.} for i in range(16)]
        final=deepcopy(base)
        for i,row in enumerate(final):row['generation']=f'final-{i}'
        final[0].update(authorship_valid=True,runtime_valid=False,games=48,banked_point_yield=8.)
        result=audit.prior.paired_summary(base,final,list(range(16)),200)
        self.assertEqual(result['mean_yield_difference'],.5)
        with self.assertRaises(ValueError):audit.prior.paired_summary(base,final[:-1],list(range(16)),200)
        invalid=deepcopy(base);invalid[0]['games']=48
        with self.assertRaises(ValueError):audit.prior.paired_summary(invalid,final,list(range(16)),200)

    def test_actual_parent_preparation_and_complete_synthetic32draw_audit(self):
        with tempfile.TemporaryDirectory(prefix='evaluation-audit-fixture-') as folder:
            run=prepared(folder)
            result=audit.audit(run)
            self.assertTrue(result['integrity_passed']);self.assertFalse(result['planned_study_complete'])
            self.assertEqual(result['inherited_RL_optimizer_steps'],3)
            backend=Backend();engine=runner.Evaluation(run,backend);backend.config=engine.config
            budget,_=runner.journal.execution_budget(engine.config);engine.status['execution_budget']=budget;engine.state()
            runtime=runner.common.load(runner.PARENT/'runtime.json')
            runtime.update(evaluation_only=True,training_allowed=False,new_optimizer_steps=0)
            engine.commit('runtime.json',runtime);engine.run_all()
            self.assertEqual(engine.status['state'],'complete')
            result=audit.audit(run)
            self.assertTrue(result['planned_study_complete']);self.assertFalse(result['parent_training_complete'])
            self.assertEqual((result['committed_trajectories'],result['committed_raw_sampled_turns']),(32,64))
            self.assertEqual(result['new_RL_optimizer_steps'],0)
            self.assertEqual(result['paired_yield']['program_cluster_n'],16)
            self.assertEqual(result['paired_yield']['mean_yield_difference'],0.)
            for label in runner.ENDPOINTS:
                self.assertTrue(result['endpoint_state_proofs'][label]['unchanged_adapter_adam_rng'])
                self.assertEqual(len(result['evaluation_cells'][label]['programs']),16)
                rows=result['evaluation_cells'][label]['programs']
                self.assertEqual(rows[0]['games'],48)
                self.assertTrue(all(row['games']==0 for row in rows[1:]))
                self.assertEqual(rows[0]['runtime_valid'],label=='base')
            calls=backend.calls;runner.Evaluation(run,backend).run_all();self.assertEqual(calls,backend.calls)
            row=runner.common.load(run/'generations/eval-cycle3-15.json');row['text']='unrecorded replacement'
            runner.common.atomic_json(run/'generations/eval-cycle3-15.json',row)
            with self.assertRaisesRegex(ValueError,'Artifact hash'):audit.audit(run)


if __name__=='__main__':unittest.main()
