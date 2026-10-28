"""Synthetic complete-report fixtures only; never inspect active evaluations."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import plot_arena_mixed_scalar as plot


def training_fixture(zero=False):
    return {'kind':'mixed_arena_scalar_independent_audit_v1','integrity_passed':True,'complete':True,
        'training_complete':True,'state':'training_complete','phase':'training','comparisons_available':False,
        'optimizer_outcome_unknown':False,'uncommitted_operations':[],
        'attempted_cohorts':20,'actual_optimizer_steps':0 if zero else 19,
        'groups':[{'cohort':i,'episodes':10,'optimizer_steps':0 if zero or i==2 else 1,
            'reward_learning_signal':False if zero else i%3!=0} for i in range(1,21)]}


def fixture(zero=False):
    metrics={e:{} for e in ('base','final')};schedule={'eval-base':[],'eval-final':[]};comparisons={}
    for j,env in enumerate(plot.ENVIRONMENTS):
        clusters=[]
        for endpoint in ('base','final'):metrics[endpoint][env]=[]
        for i in range(16):
            family=f'{env}:family-{i:02d}';a=(.2 if j>=3 else -.3)+(i%4)*.05
            delta=0. if zero else .025*j+(.04 if i%2==0 else -.02)
            combinations=[(s,o) for s in (0,1) for o in ('base','heuristic')] if j<3 else [(0,'none')]
            for endpoint in ('base','final'):
                for seat,opponent in combinations:
                    value=a+(delta if endpoint=='final' else 0.)
                    identity=f'{endpoint}-{env}-{i}-{seat}-{opponent}'
                    domain={'finish_reason':'material_adjudication_at_cap' if env=='chess' else 'distance_adjudication_at_cap',
                        'absolute_pnl':[float(i),float(-i)],'filled_units':[i+1,i],'fees':[1.,2.],
                        'shortest_distances':[i%3,(i+1)%3],'moves':[i,i+1],'checkmate':False,
                        'verified_goal':i%2==0,'turns':10,'hidden_all_passed':i%2==0,
                        'hidden_pass_fraction':i%2,'hidden_runtime_faults':0}
                    metrics[endpoint][env].append({'episode_id':identity,'family':family,
                        'learner_seat':seat,'opponent':opponent,'return':value,'metrics':domain})
                    schedule['eval-'+endpoint].append({'episode_id':identity,'environment':env,
                        'scenario_family_id':family,'learner_seats':[seat],'opponent':opponent})
            clusters.append({'scenario_family_id':family,'base':a,'final':a+delta,'delta':delta})
        comparisons[env]={'paired_clusters':16,'clusters':clusters,
            'base_mean':plot.mean([r['base'] for r in clusters]),'final_mean':plot.mean([r['final'] for r in clusters]),
            'delta':plot.mean([r['delta'] for r in clusters])}
    audit={'kind':'mixed_arena_scalar_independent_audit_v1','integrity_passed':True,'complete':True,
        'comparisons_available':True,'state':'complete','phase':'evaluation','evaluation_episodes':448,
        'optimizer_outcome_unknown':False,'uncommitted_operations':[],'actual_optimizer_steps':0 if zero else 19,
        'comparisons':comparisons,'metrics':metrics,'test_fixture':True,
        'weighted_return_delta':sum(plot.WEIGHTS[e]*comparisons[e]['delta'] for e in plot.ENVIRONMENTS)}
    cfg={'model':plot.MODEL,'revision':plot.REVISION,'environment_weights':plot.WEIGHTS,'thinking':False,
        'physical_execution_contract':deepcopy(plot.PHYSICAL_EXECUTION_CONTRACT),'likelihood_tolerance':.002}
    policies={e:{'config':cfg,'binding':{'version':0 if e=='base' else 20,'checkpoint_id':'base' if e=='base' else 'cycle20',
        'base_identity':{'fixture':True},'adapter_sha256':e+'-fixture'}} for e in ('base','final')}
    policies['_training_summary']=plot.summarize_training(training_fixture(zero))
    return audit,schedule,policies


def write_fixture(folder):
    folder=Path(folder);audit,schedule,policies=fixture();inputs={};artifacts={}
    def write(relative,value,table):
        path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));table[relative]=plot.sha(path)
    write('inputs/schedule.json',schedule,inputs)
    for e in ('base','final'):write(f'inputs/checkpoints/{e}/policy.json',policies[e],inputs)
    for e in ('base','final'):write(f'evaluation/{e}/seal.json',{'synthetic_fixture':True},artifacts)
    write('evaluation-seal.json',{'episodes':448,'paired_scenarios_per_environment':16,
        'baseline_seal_sha256':artifacts['evaluation/base/seal.json'],'final_seal_sha256':artifacts['evaluation/final/seal.json']},artifacts)
    sources={}
    for name in plot.SCALAR_SOURCES:
        source=folder/'source'/name;source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text('SYNTHETIC SOURCE '+name);sources[name]=plot.sha(source)
    parent_manifest={'kind':'mixed_arena_scalar_v1','phase':'training','sources':sources,
        'config':{'policy':policies['base']['config'],'endpoint_choices':[20],
            'episodes_per_cohort':10,'episodes_per_environment':2}}
    write('inputs/parent/manifest.json',parent_manifest,inputs)
    parent_status={'state':'training_complete','manifest_sha256':inputs['inputs/parent/manifest.json']}
    write('inputs/parent/status.json',parent_status,inputs)
    parent_audit=training_fixture();parent_audit.update(manifest_sha256=inputs['inputs/parent/manifest.json'],
        status_sha256=inputs['inputs/parent/status.json'],source_auditor_sha256=sources[plot.AUDITOR])
    write('inputs/parent/final-audit.json',parent_audit,inputs)
    completion={'attempted_cohorts':20,'actual_optimizer_steps':19,'all_updates_committed':True,
        'endpoint':{'binding':policies['final']['binding']}}
    write('inputs/parent/training-complete.json',completion,inputs)
    manifest={'kind':'mixed_arena_scalar_v1','phase':'evaluation','config':{'policy':policies['base']['config']},
        'sources':sources,'inputs':inputs}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    status={'state':'complete','claims_allowed':True,'manifest_sha256':plot.sha(folder/'manifest.json'),'artifacts':artifacts}
    (folder/'status.json').write_text(json.dumps(status))
    audit.update(manifest_sha256=plot.sha(folder/'manifest.json'),status_sha256=plot.sha(folder/'status.json'),source_auditor_sha256=sources[plot.AUDITOR])
    path=folder/'audit.json';path.write_text(json.dumps(audit));return path


class ScalarReportingTests(unittest.TestCase):
    def test_exact_cluster_denominators_and_weighted_mean_are_deterministic(self):
        args=fixture();first=plot.summarize(*args,draws=200);second=plot.summarize(*args,draws=200)
        self.assertEqual(first,second);self.assertEqual(first['evaluation_episodes'],448)
        self.assertEqual(first['actual_optimizer_steps'],19);self.assertEqual(first['attempted_cohorts'],20)
        self.assertEqual(first['training']['zero_gradient_skipped_cohorts'],1)
        self.assertEqual(first['training']['cohorts_with_nonzero_reward_advantages'],14)
        self.assertEqual(first['training']['updates_without_nonzero_reward_advantages'],6)
        self.assertEqual(first['panels']['coding']['objective_coefficient'],.04)
        self.assertEqual(first['weighted_macro_delta']['delta'],args[0]['weighted_return_delta'])
        for env in plot.ENVIRONMENTS:
            self.assertEqual(first['panels'][env]['paired_clusters'],16)
            self.assertEqual(first['panels'][env]['episodes_per_checkpoint'],64 if env in plot.ENVIRONMENTS[:3] else 16)
        self.assertEqual(first['native_metrics']['chess']['checkmate_win']['base_mean'],0.)
        self.assertEqual(first['native_metrics']['chess']['cap_rate']['base_mean'],1.)
        reordered=deepcopy(args)
        for env in plot.ENVIRONMENTS:reordered[0]['comparisons'][env]['clusters'].reverse()
        self.assertEqual(first,plot.summarize(*reordered,draws=200))

    def test_missing_native_metric_is_unavailable_not_imputed_zero(self):
        audit,schedule,policies=fixture()
        for endpoint in ('base','final'):
            for row in audit['metrics'][endpoint]['market']:row['metrics'].pop('absolute_pnl')
        native=plot.summarize(audit,schedule,policies,draws=100)['native_metrics']['market']['absolute_pnl']
        self.assertFalse(native['available']);self.assertEqual(native['observed_values'],{'base':0,'final':0})
        self.assertNotIn('base_mean',native)

    def test_partial_and_training_only_refuse_before_evidence_is_read(self):
        for key,value in (('complete',False),('comparisons_available',False),('evaluation_episodes',447),
                          ('optimizer_outcome_unknown',True),('phase','training')):
            args=list(fixture());args[0][key]=value;args[0]['metrics']='must not read'
            with self.assertRaisesRegex(ValueError,'complete'):plot.summarize(*args,draws=100)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'partial.json';path.write_text(json.dumps({'complete':False}))
            with self.assertRaisesRegex(ValueError,'complete'):plot.bind_inputs(Path(tmp)/'nonexistent',path)

    def test_prior_condition_and_wrong_physical_execution_are_rejected(self):
        args=list(fixture());args[0]['kind']='mixed_arena_independent_audit_v1'
        args[0]['metrics']='must not read'
        with self.assertRaisesRegex(ValueError,'complete'):plot.summarize(*args,draws=100)
        for change in ('manifest_kind','physical_batch','missing_scalar_source','cycle40'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                path=write_fixture(tmp);root=Path(tmp)
                manifest=json.loads((root/'manifest.json').read_text())
                if change=='manifest_kind':manifest['kind']='mixed_arena_v1'
                elif change=='physical_batch':manifest['config']['policy']['physical_execution_contract']['physical_batch_size']=8
                elif change=='missing_scalar_source':manifest['sources'].pop('selfplay/arena_mixed_scalar/envs.py')
                else:
                    policy_path=root/'inputs/checkpoints/final/policy.json';policy=json.loads(policy_path.read_text())
                    policy['binding'].update(version=40,checkpoint_id='cycle40');policy_path.write_text(json.dumps(policy))
                    manifest['inputs']['inputs/checkpoints/final/policy.json']=plot.sha(policy_path)
                (root/'manifest.json').write_text(json.dumps(manifest))
                audit=json.loads(path.read_text());status=json.loads((root/'status.json').read_text())
                status['manifest_sha256']=audit['manifest_sha256']=plot.sha(root/'manifest.json')
                (root/'status.json').write_text(json.dumps(status));audit['status_sha256']=plot.sha(root/'status.json')
                path.write_text(json.dumps(audit))
                with self.assertRaises(ValueError):plot.bind_inputs(root,path)

    def test_actual_steps_reward_signal_and_attempted_cohorts_are_distinct(self):
        for mutate in (lambda a:a['groups'].pop(),lambda a:a['groups'][0].__setitem__('optimizer_steps',2),
            lambda a:a.__setitem__('actual_optimizer_steps',20),
            lambda a:a['groups'][0].__setitem__('reward_learning_signal',1),
            lambda a:a.__setitem__('optimizer_outcome_unknown',True)):
            a=training_fixture();mutate(a)
            with self.assertRaises(ValueError):plot.summarize_training(a)
        a=training_fixture();a['groups'][0]['reward_learning_signal']=False
        x=plot.summarize_training(a)
        self.assertEqual(x['actual_optimizer_steps'],19)
        self.assertEqual(x['updates_without_nonzero_reward_advantages'],7)

    def test_role_specific_native_pnl_never_uses_other_player(self):
        for seat,expected in ((0,31.),(1,-7.)):
            row={'learner_seat':seat,'return':.2,'metrics':{'absolute_pnl':[31,-7]}}
            self.assertEqual(plot.metric_value(row,'absolute_pnl'),expected)
        self.assertEqual(plot.metric_value({'learner_seat':1,'return':-.3,'metrics':{'checkmate':True}},'checkmate_win'),0)

    def test_rejects_wrong_family_pairing_roles_returns_and_macro_denominator(self):
        changes=[lambda a:a['metrics']['base']['market'].pop(),
            lambda a:a['metrics']['base']['market'][0].__setitem__('learner_seat',1),
            lambda a:a['metrics']['final']['maze'][0].__setitem__('family','wrong'),
            lambda a:a['comparisons']['chess']['clusters'][0].__setitem__('delta',99),
            lambda a:a['metrics']['final']['planning'][0].__setitem__('return',float('nan')),
            lambda a:a.__setitem__('weighted_return_delta',999)]
        for change in changes:
            audit,schedule,policies=fixture();change(audit)
            with self.assertRaises(ValueError):plot.summarize(audit,schedule,policies,draws=100)

    def test_exact_audit_status_source_and_artifacts_bound_and_rechecked(self):
        for relative in ('audit.json','status.json','source/'+plot.AUDITOR,'inputs/schedule.json','evaluation/base/seal.json'):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory() as tmp:
                path=write_fixture(tmp);*_,bindings=plot.bind_inputs(tmp,path);plot.verify_bindings(bindings)
                target=Path(tmp)/relative;target.write_text(target.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'changed'):plot.verify_bindings(bindings)
                if relative!='audit.json':
                    with self.assertRaises(ValueError):plot.bind_inputs(tmp,path)

    def test_zero_updates_and_synthetic_publication_are_honest(self):
        data=plot.summarize(*fixture(zero=True),draws=100)
        self.assertEqual(data['weighted_macro_delta']['descriptive_95_interval'],[0.,0.])
        self.assertIn('unchanged weights',data['interpretation'])
        with tempfile.TemporaryDirectory() as tmp:
            path=write_fixture(tmp)
            with self.assertRaisesRegex(ValueError,'Synthetic'):plot.render(tmp,path,Path(tmp)/'forbidden')
            self.assertEqual(list(Path(tmp).glob('forbidden*')),[])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Optional isolated rendering runtime')
    def test_fixture_figure_and_render_race_and_evidence_change_fail_closed(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data=plot.summarize(*fixture(),draws=100)
        figure=plot.make_figure(data,plt,fixture=True)
        self.assertEqual(len(figure.axes),6)
        self.assertIn('SYNTHETIC TEST FIXTURE',' '.join(t.get_text() for t in figure.texts));plt.close(figure)
        for scenario in ('existing','mutation','race'):
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory() as tmp:
                path=write_fixture(tmp);values=list(plot.bind_inputs(tmp,path));values[0]['test_fixture']=False
                prefix=Path(tmp)/'output';target=Path(str(prefix)+'.png')
                original=plot.make_figure
                def changed(data,plt):
                    fig=original(data,plt,fixture=True)
                    if scenario=='mutation':(Path(tmp)/'status.json').write_text('{}')
                    else:target.write_bytes(b'KEEP')
                    return fig
                if scenario=='existing':target.write_bytes(b'KEEP')
                with patch.object(plot,'bind_inputs',return_value=values),patch.object(plot,'summarize',return_value=data),\
                     patch.object(plot,'make_figure',side_effect=changed),self.assertRaises((ValueError,FileExistsError)):
                    plot.render(tmp,path,prefix)
                if scenario=='mutation':self.assertEqual(list(Path(tmp).glob('output*')),[])
                else:self.assertEqual(target.read_bytes(),b'KEEP')


if __name__=='__main__':unittest.main()
