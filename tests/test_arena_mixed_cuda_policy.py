"""Real tiny CPU Torch tensors only: no pretrained model, GPU or network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from selfplay.arena_mixed import cuda_policy as policy

WEIGHTS = {"market": .24, "maze": .24, "chess": .24, "planning": .24, "coding": .04}


class Tokenizer:
    pad_token_id, eos_token_id = 0, 1
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt and enable_thinking is False
        return "<user>"+messages[0]["content"]+"<assistant><think></think>"
    def encode(self, text, add_special_tokens=False):
        return [2+ord(c)%7 for c in text]


class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})
    def forward(self, x):
        return self.lora_B["default"](self.lora_A["default"](x))


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 3)
        self.embedding.weight.requires_grad_(False)
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj, self.self_attn.v_proj = Pair(), Pair()
        self.register_buffer("packed", torch.tensor([1, 7, 255], dtype=torch.uint8))
        self.register_buffer("quant_map", torch.tensor([0., 1.]))
        self.calls = []
    def forward(self, input_ids, attention_mask, position_ids, use_cache, return_dict):
        self.calls.append({"positions": position_ids.clone(), "mask": attention_mask.clone()})
        x = self.embedding(input_ids)*attention_mask[:, :, None]
        x = x.cumsum(1)/(position_ids+1)[:, :, None]
        x = x+self.self_attn.q_proj(x)+self.self_attn.v_proj(x)
        return SimpleNamespace(last_hidden_state=x)


class Head(torch.nn.Linear):
    def __init__(self):
        super().__init__(3, 6, bias=False); self.weight.requires_grad_(False); self.shapes=[]
    def forward(self, value):
        self.shapes.append(tuple(value.shape)); return super().forward(value)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model=Core(); self.lm_head=Head()
    def get_base_model(self): return self
    def get_output_embeddings(self): return self.lm_head


def fixture(role="learner", **config):
    torch.manual_seed(77)
    obj = object.__new__(policy.CudaActionPolicy)
    obj.torch, obj.model, obj.role, obj.device = torch, Tiny(), role, "cpu"
    obj.config = policy.default_config(WEIGHTS, model_index_sha256="a"*64,
        tokenizer_files_sha256="b"*64, **config)
    obj.tokenizer, obj.labels, obj.label_ids = Tokenizer(), ["A", "B", "C", "D"], [2, 3, 4, 5]
    obj.parameters = {n:p for n,p in obj.model.named_parameters() if p.requires_grad}
    obj.optimizer = torch.optim.Adam(obj.parameters.values(), lr=obj.config["learning_rate"])
    obj.runtime = {}; obj._initialize_identity(); obj.model.eval()
    obj.base_identity = {"model":policy.MODEL, "revision":policy.REVISION,
        "model_index_sha256":"a"*64, "tokenizer_files_sha256":"b"*64,
        "quantization":"nf4-double-fp32", "frozen_state_sha256":obj._frozen["sha256"]}
    obj.binding, obj.poisoned, obj.pending_version, obj.bound_adapter = None, False, None, None
    return obj


def request(env="market", episode="e", player=0, turn=0, seed=10, actions=3, extra=""):
    return {"observation":{"state": extra or "public"}, "actions":[{"move":i} for i in range(actions)],
        "environment":env, "env_id":env, "scenario_family_id":episode,
        "episode_id":episode, "player":player, "turn":turn, "sample_seed":seed,
        "stage":"train", "cohort_id":"c0"}


def episodes(obj, unequal_lengths=False, rewards=True):
    result=[]
    for index, env in enumerate(WEIGHTS):
        seats = [0] if env in ("planning", "coding") else [0,1]
        for i in range(2):
            identifier=f"{env}-{i}"
            scores=[1.-2*i, -1.+2*i] if rewards else [0.,0.]
            if seats==[0]: scores[1]=0.
            transitions=[]
            for p in seats:
                for turn in range(2 if unequal_lengths and i==0 else 1):
                    row=obj.choose([request(env, identifier,p,turn,seed=index*100+i*10+p+turn)],obj.binding)[0]
                    row["terminal_return"]=scores[p]; transitions.append(row)
            result.append({"environment":env,"episode_id":identifier,"scenario_family_id":identifier,"learner_seats":seats,
                "returns":scores,"transitions":transitions})
    return result


class MixedCudaTests(unittest.TestCase):
    def setUp(self):
        self.cuda_rng=[torch.get_rng_state().clone()]
        self.patches=[patch.object(torch.cuda,"get_rng_state_all",side_effect=lambda:[s.clone() for s in self.cuda_rng]),
            patch.object(torch.cuda,"set_rng_state_all",side_effect=lambda values:self.cuda_rng.__setitem__(slice(None),[s.clone() for s in values]))]
        for p in self.patches: p.start()
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
    def tearDown(self):
        for p in reversed(self.patches): p.stop()
    def base(self, obj): return obj.save_checkpoint(Path(self.temp.name)/("base-"+str(id(obj))),"base",0)

    def test_mask_final_hidden_head_and_variable_length_sampling(self):
        obj=fixture(); self.base(obj)
        requests=[request(seed=12,actions=2),request(seed=33,actions=4,extra="longer public description"*5)]
        combined=obj.choose(requests,obj.binding)
        for r, actual in zip(requests,combined):
            single=obj.choose([r],obj.binding)[0]
            self.assertEqual(actual["action_index"],single["action_index"])
            self.assertTrue(torch.allclose(torch.tensor(actual["old_log_probs"]),torch.tensor(single["old_log_probs"]),atol=1e-6))
            self.assertEqual(len(actual["old_log_probs"]),len(r["actions"]))
            self.assertAlmostEqual(sum(map(__import__('math').exp, actual["old_log_probs"])),1.,places=6)
        self.assertTrue(all(len(shape)==2 and shape[-1]==3 for shape in obj.model.lm_head.shapes))
        masks=obj.model.model.calls[0]
        self.assertTrue(torch.equal(masks["positions"][0,:masks["mask"][0].sum()],torch.arange(masks["mask"][0].sum())))

    def test_exact_sampling_and_request_provenance_reject_tampering(self):
        obj=fixture(); self.base(obj); row=obj.choose([request()],obj.binding)[0]
        obj._validate_row(row,obj.binding)
        for key,value in (("input_ids",[3]),("menu_sha256","f"*64),("action_token_id",99),
                          ("old_log_probs",[-1.,-1.,-1.]),("prompt","hidden reward")):
            bad=deepcopy(row);bad[key]=value
            with self.assertRaises(ValueError):obj._validate_row(bad,obj.binding)
        bad=deepcopy(obj.binding);bad["adapter_sha256"]="f"*64
        with self.assertRaises(ValueError):obj.choose([request()],bad)

    def test_durable_json_roundtrip_preserves_prompt_tokens_and_all_hashes(self):
        obj=fixture();self.base(obj)
        row=request()
        row['observation']={'z':'first insertion','role_count':2,
            'accounts':{10:{'z':1,'a':2},2:'two',-1:'external'},'a':'last insertion'}
        row['actions']=[{'z':i,'a':{'second':2,'first':1}} for i in range(3)]
        sampled=obj.choose([row],obj.binding)[0]
        restored=json.loads(policy.canonical(sampled))
        obj._validate_row(restored,obj.binding)
        repeated=obj.choose([json.loads(policy.canonical(row))],obj.binding)[0]
        for key in ('prompt','rendered_prompt','input_ids','input_ids_sha256',
                    'prompt_sha256','menu_sha256','old_log_probs','action_index'):
            self.assertEqual(sampled[key],repeated[key],key)
        self.assertEqual(policy.digest(sampled),policy.digest(restored))

    def test_weighted_roles_loo_and_no_realized_length_divisor(self):
        obj=fixture();self.base(obj); data=episodes(obj,unequal_lengths=True)
        rows=policy.cohort_rows(data,WEIGHTS)
        for row in rows:
            seats=1 if row["environment"] in ("planning","coding") else 2
            self.assertEqual(row["weight"],WEIGHTS[row["environment"]]/2/seats)
            self.assertEqual(abs(row["advantage"]),2.)
        short=next(r for r in rows if r["episode_id"]=="market-1")
        long=next(r for r in rows if r["episode_id"]=="market-0")
        self.assertEqual(short["weight"],long["weight"])
        bad=deepcopy(data);bad[0]["transitions"][0]["terminal_return"]=0
        with self.assertRaises(ValueError):policy.cohort_rows(bad,WEIGHTS)
        with self.assertRaises(ValueError):policy.cohort_rows(data[:-2],WEIGHTS)

    def test_objective_direction_clipping_and_illegal_label_mask(self):
        cfg=policy.default_config(WEIGHTS,entropy_coefficient=0.,kl_coefficient=0.)
        for advantage,direction in ((1.,-1),(-1.,1)):
            logits=torch.tensor([[0.,0.,-1e9]],requires_grad=True)
            lp=logits.log_softmax(-1); old=lp.detach().clone()
            loss,_=policy.objective(lp,old,torch.tensor([2]),torch.tensor([0]),torch.tensor([advantage]),torch.tensor([.24]),cfg)
            loss.backward();self.assertEqual(int(torch.sign(logits.grad[0,0])),direction)
            self.assertEqual(float(logits.grad[0,2]),0.)
        logits=torch.tensor([[__import__('math').log(.8),__import__('math').log(.2)]],requires_grad=True)
        old=torch.tensor([[-__import__('math').log(2)]*2])
        loss,_=policy.objective(logits.log_softmax(-1),old,torch.tensor([2]),torch.tensor([0]),torch.tensor([1.]),torch.tensor([1.]),cfg)
        loss.backward();self.assertEqual(float(logits.grad.abs().max()),0.)

    def test_gate_all_vectors_and_one_accumulated_persistent_adam_step(self):
        obj=fixture();self.base(obj); data=episodes(obj); binding=deepcopy(obj.binding)
        raw=[r for e in data for r in e["transitions"]]
        gate=obj.full_likelihood_gate(raw,binding)
        self.assertTrue(gate["passed"]);self.assertEqual(len(gate["records"]),2*len(raw))
        report=obj.train_cohort(data,binding,gate)
        self.assertEqual(report["optimizer_steps"],1);self.assertEqual(report["adam_step_after"],1)
        self.assertGreater(report["parameter_delta_l2"],0);self.assertIsNone(obj.binding)
        self.assertEqual(len(report['learner_records']),len(raw))
        self.assertEqual(report['learner_records'][0]['row_sha256'],policy.digest(raw[0]))
        reconstructed=0.
        for record,row in zip(report['learner_records'],raw):
            a=record['learner_log_probs'];old=row['old_log_probs'];i=row['action_index'];adv=record['advantage']
            math=__import__('math');ratio=math.exp(a[i]-old[i])
            entropy=-sum(math.exp(x)*x for x in a)
            kl=sum(math.exp(x)*(x-y) for x,y in zip(old,a))
            reconstructed+=record['weight']*(-min(ratio*adv,max(.8,min(1.2,ratio))*adv)-obj.config['entropy_coefficient']*entropy+obj.config['kl_coefficient']*kl)
        self.assertAlmostEqual(reconstructed,report['loss'],places=6)
        with self.assertRaises(ValueError):obj.choose([request()],binding)
        checkpoint=obj.save_checkpoint(Path(self.temp.name)/"cycle1","cycle1",1)
        self.assertEqual(checkpoint["step"],1)
        # Second cohort is sampled from fixed base while learner is one version newer.
        report2=obj.train_cohort(data,binding,gate)
        self.assertEqual(report2["policy_lag"],1);self.assertEqual(report2["adam_step_after"],2)
        obj.save_checkpoint(Path(self.temp.name)/"cycle2","cycle2",2)
        with self.assertRaisesRegex(ValueError,"lag"):obj.train_cohort(data,binding,gate)

    def test_gate_wrong_checkpoint_vector_and_missing_coverage_fail_closed(self):
        obj=fixture();self.base(obj);data=episodes(obj);raw=[r for e in data for r in e["transitions"]]
        gate=obj.full_likelihood_gate(raw,obj.binding)
        for change in (lambda x:x["records"].pop(),lambda x:x["records"][0]["actual_log_probs"].__setitem__(0,-99.),
                       lambda x:x["binding"].__setitem__("adapter_sha256","f"*64)):
            bad=deepcopy(gate);change(bad)
            with self.assertRaises(ValueError):obj.train_cohort(data,obj.binding,bad)
        with torch.no_grad():next(iter(obj.parameters.values())).add_(.5)
        with self.assertRaisesRegex(ValueError,"Live adapter"):obj.full_likelihood_gate(raw,obj.binding)

    def test_checkpoint_restore_optimizer_rng_and_behavior_gate_restore(self):
        obj=fixture();base=self.base(obj);base_folder=Path(self.temp.name)/("base-"+str(id(obj)))
        data=episodes(obj);raw=[r for e in data for r in e["transitions"]];behavior=deepcopy(obj.binding)
        gate=obj.full_likelihood_gate(raw,behavior);obj.train_cohort(data,behavior,gate)
        folder=Path(self.temp.name)/"cycle1";saved=obj.save_checkpoint(folder,"cycle1",1)
        parameters={n:p.detach().clone() for n,p in obj.parameters.items()}
        adam=obj._optimizer_identity();rng=torch.get_rng_state().clone();binding=deepcopy(obj.binding)
        stale_gate=obj.gate_at_checkpoint(raw,base_folder,base["files"])
        self.assertTrue(stale_gate["passed"]);self.assertEqual(obj.binding,binding)
        self.assertEqual(obj._optimizer_identity(),adam);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertTrue(all(torch.equal(p,parameters[n]) for n,p in obj.parameters.items()))
        actor=fixture("actor");actor.load_checkpoint(folder,saved["files"])
        self.assertEqual(actor._optimizer_identity(),adam)
        with self.assertRaisesRegex(ValueError,"Actor"):actor.train_cohort(data,behavior,stale_gate)
        self.assertEqual(policy.checkpoint_state(folder)["files"],saved["files"])
        with torch.no_grad():obj.model.model.packed[0]+=1
        with self.assertRaisesRegex(ValueError,"Frozen"):obj.verify_frozen()

    def test_zero_signal_regularizer_is_honest_and_zero_gradient_can_skip(self):
        obj=fixture(entropy_coefficient=0.,kl_coefficient=0.);self.base(obj)
        data=episodes(obj,rewards=False);raw=[r for e in data for r in e["transitions"]]
        report=obj.train_cohort(data,deepcopy(obj.binding),obj.full_likelihood_gate(raw,obj.binding))
        self.assertFalse(report["reward_learning_signal"]);self.assertEqual(report["optimizer_steps"],0)
        self.assertEqual(obj.save_checkpoint(Path(self.temp.name)/"skip","cycle1",1)["step"],0)
        regularized=fixture();self.base(regularized);data=episodes(regularized,rewards=False)
        raw=[r for e in data for r in e["transitions"]]
        report=regularized.train_cohort(data,deepcopy(regularized.binding),regularized.full_likelihood_gate(raw,regularized.binding))
        self.assertFalse(report["reward_learning_signal"]);self.assertEqual(report["optimizer_steps"],1)
        self.assertNotEqual(report["entropy_loss"],0.)

    def test_failed_optimizer_is_poisoned_and_never_retried(self):
        obj=fixture();self.base(obj);data=episodes(obj);binding=deepcopy(obj.binding)
        raw=[r for e in data for r in e["transitions"]];gate=obj.full_likelihood_gate(raw,binding)
        original=obj.optimizer.step
        def broken(): original();raise RuntimeError("ambiguous after step")
        with patch.object(obj.optimizer,"step",side_effect=broken):
            with self.assertRaisesRegex(RuntimeError,"ambiguous"):obj.train_cohort(data,binding,gate)
        self.assertTrue(obj.poisoned);self.assertIsNone(obj.last_failure["optimizer_steps"])
        with self.assertRaises(ValueError):obj.save_checkpoint(Path(self.temp.name)/"bad","cycle1",1)

    def test_smoke_preserves_all_state_and_compares_two_independent_copies(self):
        obj=fixture();base=self.base(obj);folder=Path(self.temp.name)/("base-"+str(id(obj)))
        for p in obj.parameters.values():p.grad=torch.ones_like(p)*.125
        before=obj._state_identity();requests=[request(seed=71),request(seed=83,extra="different length")]
        smoke=obj.numerical_smoke(requests)
        self.assertTrue(smoke["passed"],smoke.get("reason"));self.assertEqual(before,obj._state_identity())
        actor=fixture("actor");actor.load_checkpoint(folder,base["files"])
        actor_report=actor.numerical_smoke(requests)
        compared=policy.compare_actor_reports(actor_report,smoke)
        self.assertTrue(compared["passed"]);self.assertEqual(smoke["optimizer_steps"],0)
        bad=deepcopy(actor_report);bad["rows"][0]["old_log_probs"][0]-=.1
        self.assertFalse(policy.compare_actor_reports(bad,smoke)["passed"])

    def test_paired_scenario_baseline_rejected_and_local_input_hashes_exact(self):
        obj=fixture();self.base(obj);data=episodes(obj)
        data[1]["scenario_family_id"]=data[0]["scenario_family_id"]
        with self.assertRaisesRegex(ValueError,"scenario families"):policy.cohort_rows(data,WEIGHTS)
        folder=Path(self.temp.name)/"input";folder.mkdir()
        for name in ("model.safetensors.index.json","tokenizer.json","tokenizer_config.json"):(folder/name).write_text(name)
        first=policy.local_input_identity(folder);(folder/"tokenizer_config.json").write_text("changed")
        self.assertNotEqual(first["tokenizer_files_sha256"],policy.local_input_identity(folder)["tokenizer_files_sha256"])

    def test_single_player_prompt_and_original_label_enumeration(self):
        text=policy.action_prompt({"role_count":1,"state":"public"},[{"tool":"move"}],["A"])
        self.assertIn("single-player",text);self.assertNotIn("opponent",text)
        self.assertIn("opponent",policy.action_prompt({"role_count":2},[{"move":1}],["A"]))
        class Labels:
            def get_vocab(self):return {"AA":0,"AB":1}
            def encode(self,text,add_special_tokens):return [ord(text)] if len(text)==1 else [77,88]
            def decode(self,ids):return chr(ids[0])
        self.assertEqual(policy.select_labels(Labels(),4),(["A","B","C","D"],[65,66,67,68]))

    def test_full_cohort_profile_performs_real_temporary_adam_then_restores_everything(self):
        obj=fixture();self.base(obj);data=episodes(obj);binding=deepcopy(obj.binding)
        raw=[r for e in data for r in e["transitions"]];gate=obj.full_likelihood_gate(raw,binding)
        for p in obj.parameters.values():p.grad=torch.ones_like(p)*.375
        before=obj._state_identity()
        profile_folder=Path(self.temp.name)/'profile-checkpoint'
        report=obj.profile_cohort(data,binding,gate,profile_folder)
        self.assertTrue(report["passed"]);self.assertEqual(report["temporary_optimizer_steps"],1)
        self.assertEqual(report["retained_optimizer_steps"],0);self.assertEqual(obj._state_identity(),before)
        self.assertGreater(report["temporary_update"]["parameter_delta_l2"],0.)
        inspected=policy.checkpoint_state(profile_folder)
        self.assertEqual(inspected['step'],1)
        self.assertEqual(report['temporary_checkpoint']['files'],inspected['files'])
        self.assertGreaterEqual(report['checkpoint_seconds'],0.)
        self.assertGreaterEqual(report['elapsed_seconds'],report['checkpoint_seconds'])
        self.assertNotEqual(inspected['policy']['binding']['adapter_sha256'],binding['adapter_sha256'])
        original=obj.optimizer.step
        def broken():original();raise RuntimeError("profile ambiguity")
        with patch.object(obj.optimizer,"step",side_effect=broken):report=obj.profile_cohort(data,binding,gate)
        self.assertFalse(report["passed"]);self.assertTrue(report["state_restored"]);self.assertTrue(obj.poisoned)
        with self.assertRaises(ValueError):obj.profile_cohort(data,binding,gate)

    def test_accumulated_microbatches_match_full_objective_and_adam(self):
        obj=fixture();base=self.base(obj);folder=Path(self.temp.name)/("base-"+str(id(obj)))
        data=episodes(obj,unequal_lengths=True);binding=deepcopy(obj.binding)
        raw=[r for e in data for r in e["transitions"]];gate=obj.full_likelihood_gate(raw,binding)
        expected=fixture();expected.load_checkpoint(folder,base["files"])
        rows=policy.cohort_rows(data,WEIGHTS);lp=expected.log_probs(rows)
        old=torch.tensor([r["old_log_probs"]+[-1e9]*(len(expected.labels)-len(r["old_log_probs"])) for r in rows])
        loss,_=policy.objective(lp,old,torch.tensor([len(r["actions"]) for r in rows]),
            torch.tensor([r["action_index"] for r in rows]),torch.tensor([r["advantage"] for r in rows]),
            torch.tensor([r["weight"] for r in rows]),expected.config)
        loss.backward();torch.nn.utils.clip_grad_norm_(list(expected.parameters.values()),expected.config["gradient_clip"])
        expected.optimizer.step()
        report=obj.train_cohort(data,binding,gate)
        self.assertAlmostEqual(report["loss"],float(loss.detach()),places=6)
        for n,p in obj.parameters.items():self.assertTrue(torch.allclose(p,expected.parameters[n],atol=1e-7,rtol=1e-6),n)


if __name__=="__main__": unittest.main()
