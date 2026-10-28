"""Synthetic complete-report fixtures only; never inspect active evaluations."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import plot_arena_mixed as plot


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
    audit={'kind':'mixed_arena_independent_audit_v1','integrity_passed':True,'complete':True,
        'comparisons_available':True,'state':'complete','phase':'evaluation','evaluation_episodes':448,
        'optimizer_outcome_unknown':False,'uncommitted_operations':[],'actual_optimizer_steps':0 if zero else 19,
        'comparisons':comparisons,'metrics':metrics,'test_fixture':True,
        'weighted_return_delta':sum(plot.WEIGHTS[e]*comparisons[e]['delta'] for e in plot.ENVIRONMENTS)}
    cfg={'model':plot.MODEL,'revision':plot.REVISION,'environment_weights':plot.WEIGHTS,'thinking':False}
    policies={e:{'config':cfg,'binding':{'version':0 if e=='base' else 20,'checkpoint_id':'base' if e=='base' else 'cycle20',
        'base_identity':{'fixture':True},'adapter_sha256':e+'-fixture'}} for e in ('base','final')}
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
    source=folder/'source'/plot.AUDITOR;source.parent.mkdir(parents=True);source.write_text('SYNTHETIC AUDITOR')
    manifest={'kind':'mixed_arena_v1','phase':'evaluation','config':{'policy':policies['base']['config']},
        'sources':{plot.AUDITOR:plot.sha(source)},'inputs':inputs}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    status={'state':'complete','claims_allowed':True,'manifest_sha256':plot.sha(folder/'manifest.json'),'artifacts':artifacts}
    (folder/'status.json').write_text(json.dumps(status))
    audit.update(manifest_sha256=plot.sha(folder/'manifest.json'),status_sha256=plot.sha(folder/'status.json'),source_auditor_sha256=plot.sha(source))
    path=folder/'audit.json';path.write_text(json.dumps(audit));return path


class MixedReportingTests(unittest.TestCase):
    def test_exact_cluster_denominators_and_weighted_mean_are_deterministic(self):
        args=fixture();first=plot.summarize(*args,draws=200);second=plot.summarize(*args,draws=200)
        self.assertEqual(first,second);self.assertEqual(first['evaluation_episodes'],448)
        self.assertEqual(first['actual_optimizer_steps'],19);self.assertEqual(first['attempted_cohorts'],20)
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
