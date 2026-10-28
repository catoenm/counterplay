"""Native low-thinking drafts and native-disabled revisions, no optimization.

Sampling retains independent row generators and the frozen FP32/NF4 arithmetic.
The mixed-mode derivative smoke and full sixteen-turn likelihood replay restore
RNG, gradients, configuration and model mode and verify unchanged adapter/Adam.
No pretrained model is loaded on import.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

from .program_debug_calibration_cuda import CalibrationCudaPolicy
from .program_cardinal_cuda import tensor_identity
from scripts.cloud import train_program_rl as common

MAX_BATCH_SIZE = 2
STAGES = ('draft', 'revision')


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def stage_metadata(stage):
    if stage not in STAGES:
        raise ValueError('Explicit authoring_stage draft or revision required')
    return {'authoring_stage': stage, 'thinking_enabled': stage == 'draft',
            'reasoning_effort': 'low' if stage == 'draft' else None,
            'reasoning_mode': 'native_low' if stage == 'draft' else 'native_disabled'}


class RevisionCudaPolicy(CalibrationCudaPolicy):
    def template(self, messages, authoring_stage='draft'):
        mode = stage_metadata(authoring_stage)
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        if (not isinstance(messages, list) or not messages
                or any(set(m) != {'role', 'content'} or m['role'] not in ('user', 'assistant')
                       or not isinstance(m['content'], str) for m in messages)):
            raise ValueError('Explicit canonical user/assistant messages required')
        kwargs = {'enable_thinking': mode['thinking_enabled']}
        if mode['thinking_enabled']:
            kwargs['reasoning_effort'] = 'low'
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True, **kwargs)
        if mode['thinking_enabled']:
            if (not rendered.endswith('<|im_start|>assistant\n<think>\n')
                    or not rendered.startswith('<|im_start|>system\nReasoning effort is set to low.')):
                raise ValueError('Native draft template did not enable explicit low reasoning')
        elif (not rendered.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
              or not rendered.startswith('<|im_start|>user\n')):
            raise ValueError('Native revision template did not close its empty thinking prefix')
        return rendered, self.tokenizer.encode(rendered, add_special_tokens=False)

    def sample_batch(self, prompt, seeds, authoring_stage='draft'):
        contexts = prompt['batch_messages'] if isinstance(prompt, dict) else [prompt] * len(seeds)
        return self.sample_messages_batch(contexts, seeds, authoring_stage=authoring_stage)

    def sample_messages_batch(self, contexts, seeds, authoring_stage='draft'):
        stage_metadata(authoring_stage)
        return self._sample_native_batch(contexts, seeds, [authoring_stage] * len(seeds))

    def _sample_native_batch(self, contexts, seeds, stages):
        torch, cfg = self.torch, self.config
        if (not isinstance(seeds, (tuple, list)) or not 1 <= len(seeds) <= MAX_BATCH_SIZE
                or len(contexts) != len(seeds) or len(stages) != len(seeds)
                or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)):
            raise ValueError('One or two explicit contexts and independent seeds required')
        modes = [stage_metadata(stage) for stage in stages]
        templates = [self.template(messages, stage) for messages, stage in zip(contexts, stages)]
        if any(not ids or len(ids) + cfg['max_new_tokens'] > cfg['max_context_tokens'] for _, ids in templates):
            raise ValueError('Prompt plus complete response reserve exceeds context')
        device = getattr(self, '_sample_device', 'cuda')
        width = max(len(ids) for _, ids in templates)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.eos[0]
        ids = torch.tensor([[pad] * (width-len(row)) + row for _, row in templates], dtype=torch.long, device=device)
        mask = torch.tensor([[0] * (width-len(row)) + [1] * len(row) for _, row in templates], dtype=torch.long, device=device)
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        sampled, logps = [[] for _ in seeds], [[] for _ in seeds]
        finish, elapsed = ['length'] * len(seeds), [None] * len(seeds)
        active, counts, cache = list(range(len(seeds))), [], None
        started = time.monotonic()
        self.model.eval()
        with torch.no_grad():
            for _ in range(cfg['max_new_tokens']):
                if time.monotonic()-started >= cfg['sample_seconds']:
                    for i in active:
                        finish[i] = 'time'
                    break
                counts.append(len(active))
                positions = (mask.cumsum(-1)-1).masked_fill(mask == 0, 0)[:, -ids.shape[1]:]
                output = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                lp = torch.log_softmax(output.logits[:, -1].float(), dim=-1)
                if not bool(torch.isfinite(lp[active]).all()):
                    raise ValueError('Nonfinite sampled distribution')
                chosen = torch.full((len(seeds),), self.eos[0], dtype=torch.long, device=device)
                for i in active:
                    chosen[i] = torch.multinomial(lp[i].exp(), 1, generator=generators[i])[0]
                values = lp.gather(1, chosen[:, None]).squeeze(1).cpu().tolist()
                tokens, cache = chosen.cpu().tolist(), output.past_key_values
                del output, lp
                remaining = []
                for i in active:
                    sampled[i].append(tokens[i]); logps[i].append(float(values[i]))
                    if tokens[i] in self.eos:
                        finish[i], elapsed[i] = 'eos', time.monotonic()-started
                    else:
                        remaining.append(i)
                active = remaining
                if not active:
                    break
                # Finished rows continue as fixed-cache placeholders only. They
                # never consume further RNG draws or sampled completion tokens.
                ids = chosen[:, None]
                mask = torch.cat([mask, torch.ones((len(seeds), 1), dtype=mask.dtype, device=device)], dim=1)
        del cache
        duration, records = time.monotonic()-started, []
        for i, seed in enumerate(seeds):
            rendered, prefix = templates[i]
            records.append({'text': self.tokenizer.decode(sampled[i], skip_special_tokens=True),
                'raw_text': self.tokenizer.decode(sampled[i], skip_special_tokens=False),
                'messages': contexts[i], 'prompt_template': rendered, 'prompt_ids': prefix,
                'completion_ids': sampled[i], 'sampled_token_logprobs': logps[i],
                'seed': seed, 'finish_reason': finish[i], 'input_tokens': len(prefix),
                'output_tokens': len(sampled[i]), 'elapsed_seconds': elapsed[i] or duration,
                'eos_token_ids': list(self.eos), 'sampling_batch_size': len(seeds),
                'sampling_batch_row': i, 'left_padding_tokens': width-len(prefix),
                'post_eos_cache_forwards': len(counts)-len(sampled[i]) if finish[i]=='eos' else 0,
                'completion_token_limit': cfg['max_new_tokens'], 'calibration_only': True, **modes[i]})
        return {'records': records, 'timing': {'batch_elapsed_seconds': duration,
            'total_output_tokens': sum(map(len, sampled)), 'active_sequences': len(seeds),
            'active_sequences_by_step': counts, 'batch_steps': len(counts),
            'finished_eos': finish.count('eos'), 'constant_cache_batch_size': len(seeds),
            'authoring_stages': list(stages)}}

    def validate_record(self, record):
        expected = stage_metadata(record.get('authoring_stage'))
        if any(record.get(k) != v or type(record.get(k)) is not type(v) for k, v in expected.items()):
            raise ValueError('Sampled authoring stage/thinking metadata differs from native mode')
        rendered, ids = self.template(record['messages'], expected['authoring_stage'])
        if record.get('prompt_template') != rendered or record.get('prompt_ids') != ids:
            raise ValueError('Sampled native template or prompt token IDs differ')
        completion, sampled = record['completion_ids'], record['sampled_token_logprobs']
        if (record['input_tokens'] != len(ids) or record['output_tokens'] != len(completion)
                or len(completion) != len(sampled) or not completion
                or len(ids)+record['completion_token_limit'] > self.config['max_context_tokens']
                or record['completion_token_limit'] != self.config['max_new_tokens']
                or len(completion) > record['completion_token_limit']
                or any(type(x) is not int or x < 0 for x in completion)
                or any(type(x) not in (int, float) or not math.isfinite(x) or x > 1e-6 for x in sampled)):
            raise ValueError('Completion likelihood coverage, limits or values differ')
        if (record['finish_reason'] not in ('eos', 'length', 'time')
                or (record['finish_reason']=='length' and len(completion)!=record['completion_token_limit'])):
            raise ValueError('Finish reason differs from sampled completion limit')
        if (record.get('eos_token_ids') != list(self.eos)
                or any(x in self.eos for x in completion[:-1])
                or ((completion[-1] in self.eos) != (record['finish_reason']=='eos'))):
            raise ValueError('Recorded completion includes post-EOS tokens or incorrect finish metadata')
        if (record['text'] != self.tokenizer.decode(completion, skip_special_tokens=True)
                or record['raw_text'] != self.tokenizer.decode(completion, skip_special_tokens=False)):
            raise ValueError('Sampled decoded text differs from original completion tokens')
        return True

    def token_logps(self, record):
        self.validate_record(record)
        return super().token_logps(record)

    def _capture(self):
        return {'config': self.config, 'cpu_rng': self.torch.get_rng_state().clone(),
            'cuda_rng': [x.clone() for x in self.torch.cuda.get_rng_state_all()],
            'mode': self.model.training,
            'grads': {n: None if p.grad is None else p.grad.detach().clone() for n,p in self.parameters.items()},
            'identity': self._state_identity()}

    def _state_identity(self):
        return {'adapter_sha256': object_hash({n:tensor_identity(p) for n,p in self.parameters.items()}),
            'optimizer_state_sha256': self._optimizer_identity()['state_sha256'],
            'rng': {'cpu':tensor_identity(self.torch.get_rng_state()),
                    'cuda':[tensor_identity(x) for x in self.torch.cuda.get_rng_state_all()]},
            'gradients_sha256': object_hash({n:None if p.grad is None else tensor_identity(p.grad) for n,p in self.parameters.items()}),
            'config_sha256': object_hash(self.config), 'model_training': self.model.training,
            'frozen_state_sha256': self.verify_frozen()}

    def _restore(self, saved, report):
        self.config = saved['config']
        for name, parameter in self.parameters.items():
            parameter.grad = saved['grads'][name]
        self.model.train(saved['mode'])
        self.torch.set_rng_state(saved['cpu_rng'])
        self.torch.cuda.set_rng_state_all(saved['cuda_rng'])
        report['state_before'] = saved['identity']
        try:
            after = self._state_identity(); report['state_after'] = after
            keys = {'adapter_unchanged':'adapter_sha256','optimizer_unchanged':'optimizer_state_sha256',
                'rng_restored':'rng','gradients_restored':'gradients_sha256','config_restored':'config_sha256',
                'model_mode_restored':'model_training','frozen_state_unchanged':'frozen_state_sha256'}
            for flag,key in keys.items():report[flag]=saved['identity'][key]==after[key]
            if not all(report[k] for k in keys):report.update(passed=False,reason='Likelihood admission mutated state')
        except Exception as error:
            report.update(passed=False,restoration_error_type=type(error).__name__,restoration_error=str(error)[:2000])

    def _replay_vectors(self, records, derivative=False):
        torch, reports = self.torch, []
        if derivative:
            self.optimizer.zero_grad(set_to_none=True)
        for index, record in enumerate(records):
            entry = {'index':index,'record_sha256':object_hash(record),
                'authoring_stage':record.get('authoring_stage'),'seed':record.get('seed'),
                'completion_tokens':len(record['completion_ids']),
                'sampled_token_logprobs':list(record['sampled_token_logprobs'])}
            for mode in ('eval','train'):
                values = None
                try:
                    self.validate_record(record)
                    self.model.train(mode=='train')
                    with torch.enable_grad() if mode=='train' else torch.no_grad():
                        entry[mode+'_grad_enabled'] = torch.is_grad_enabled()
                        values = self.token_logps(record)
                        vector = values.detach().cpu().tolist()
                        finite = all(math.isfinite(v) for v in vector)
                        entry[mode+'_token_logprobs'] = [v if math.isfinite(v) else str(v) for v in vector]
                        entry[mode+'_all_finite'] = finite
                        check = common.check_likelihoods(record['sampled_token_logprobs'], vector,
                            self.config['token_logprob_tolerance'], self.config['sequence_logprob_tolerance'])
                        entry[mode+'_check'] = check
                        if mode=='train' and derivative and check['passed']:
                            (-values.sum()/len(records)).backward()
                except Exception as error:
                    entry.setdefault(mode+'_token_logprobs',None)
                    entry.setdefault(mode+'_all_finite',False)
                    entry[mode+'_check']={'passed':False,'error_type':type(error).__name__,'reason':str(error)[:2000]}
                finally:
                    del values
            entry['passed'] = all(entry[m+'_check']['passed'] and entry[m+'_all_finite'] for m in ('eval','train'))
            reports.append(entry)
        result={'records':reports,'passed':all(r['passed'] for r in reports)}
        if derivative:
            squared,missing,finite=0.0,[],True
            for name,p in self.parameters.items():
                if p.grad is None:missing.append(name)
                elif not bool(torch.isfinite(p.grad).all()):finite=False
                else:squared+=float(p.grad.detach().double().square().sum().cpu())
            result['derivative']={'gradient_l2_norm':math.sqrt(squared) if math.isfinite(squared) else None,
                'missing_gradient_names':missing,'finite_nonzero':finite and math.isfinite(squared) and squared>0 and not missing}
            result['passed'] &= result['derivative']['finite_nonzero']
        return result

    def mechanics_smoke(self, prompt):
        contexts = prompt['batch_messages'] if isinstance(prompt,dict) else [prompt]*2
        if len(contexts)!=2:raise ValueError('Mixed-native smoke requires exactly two contexts')
        saved=self._capture()
        report={'passed':False,'kind':'revision_mixed_native_derivative_admission','training_use':False,
            'optimizer_steps':0,'batch_size':2,'max_new_tokens':min(32,self.config['max_new_tokens']),
            'authoring_stages':['draft','revision'],'seeds':[70_000_000+self.config['seed']+i for i in range(2)],
            'token_tolerance':self.config['token_logprob_tolerance'],
            'sequence_tolerance':self.config['sequence_logprob_tolerance']}
        try:
            self.config={**self.config,'max_new_tokens':report['max_new_tokens']}
            batch=self._sample_native_batch(contexts,report['seeds'],report['authoring_stages'])
            report['batch']=batch
            report.update(self._replay_vectors(batch['records'],derivative=True))
        except Exception as error:
            report.update(passed=False,error_type=type(error).__name__,reason=str(error)[:2000])
        finally:self._restore(saved,report)
        return report

    numerical_smoke=mechanics_smoke

    def full_completion_likelihood_gate(self, records):
        saved=self._capture()
        report={'passed':False,'kind':'revision_full_completion_likelihood_admission','training_use':False,
            'optimizer_steps':0,'expected_records':16,'token_tolerance':self.config['token_logprob_tolerance'],
            'sequence_tolerance':self.config['sequence_logprob_tolerance']}
        try:
            if (not isinstance(records,list) or len(records)!=16
                    or [r.get('authoring_stage') for r in records].count('draft')!=8
                    or [r.get('authoring_stage') for r in records].count('revision')!=8):
                raise ValueError('Full likelihood gate requires all sixteen original turns, eight per stage')
            report.update(self._replay_vectors(records,derivative=False))
            report['sampled_tokens']=sum(len(r['completion_ids']) for r in records)
        except Exception as error:
            report.update(passed=False,error_type=type(error).__name__,reason=str(error)[:2000])
        finally:self._restore(saved,report)
        return report
