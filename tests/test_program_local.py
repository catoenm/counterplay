"""CPU-only generation-protocol tests: no MLX imports or model loading."""
import json
from pathlib import Path
import tempfile
import unittest

from selfplay import program_local as local


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        self.messages,self.kwargs=messages,kwargs
        return 'user:'+messages[0]['content']+'\nassistant:'+('<think>\n' if kwargs['enable_thinking'] else '<think>\n\n</think>\n')
    def encode(self,text,**kwargs):
        self.encode_kwargs=kwargs
        return list(text.encode())


def manifest(thinking=True):
    return {'model':{'model':'fake-local-model','revision':'a'*40},'prompt':'Write a controller.',
        'prompt_sha256':'fake-prompt-hash','thinking':thinking,'samples':2,'max_tokens':4096,
        'max_seconds':0.3,'seeds':[1,2],'sampling':dict(local.SAMPLING),'source_sha256':'fake-source-hash'}


def row(text,token=1,count=1,finish=None):
    return {'text':text,'token':token,'generation_tokens':count,'prompt_tokens':5,'finish_reason':finish}


class ProgramLocalTests(unittest.TestCase):
    def test_formatter_toggles_only_thinking_and_avoids_special_token_duplication(self):
        tokenizer=Tokenizer()
        for thinking in (False,True):
            text,ids=local.format_prompt(tokenizer,'Same prompt.',thinking)
            self.assertEqual(tokenizer.messages,[{'role':'user','content':'Same prompt.'}])
            self.assertEqual(tokenizer.kwargs,{'tokenize':False,'add_generation_prompt':True,'enable_thinking':thinking})
            self.assertEqual(tokenizer.encode_kwargs,{'add_special_tokens':False})
            self.assertEqual(ids,list(text.encode()))

    def test_missing_thinking_close_never_exposes_python(self):
        thoughts='<think>Maybe implement this:\n```python\ndef act(obs): return 0\n```'
        for thinking in (False,True):
            result=local.extract_final_answer(thoughts,thinking)
            self.assertEqual(result['final_answer'],'')
            self.assertEqual(result['extraction_status'],'unfinished_reasoning')
        # Qwen's opening tag can already belong to the prompt template.
        result=local.extract_final_answer('def act(obs): return 0',True)
        self.assertEqual(result['final_answer'],'')

    def test_closed_thinking_and_non_thinking_return_only_final_answer(self):
        code='```python\ndef act(obs, legal_actions, memory):\n    return legal_actions[0], memory\n```'
        result=local.extract_final_answer('Planning code.\n</think>\n'+code,True)
        self.assertEqual(result['final_answer'],code)
        self.assertEqual(result['reasoning'],'Planning code.')
        self.assertTrue(result['reasoning_closed'])
        self.assertEqual(local.extract_final_answer(code,False)['final_answer'],code)
        self.assertEqual(local.extract_final_answer('<think>A</think>\n'+code,False)['final_answer'],code)

    def test_unexpected_or_reopened_thinking_is_rejected(self):
        for text,thinking in [('oops</think>code',False),('a</think>code<think>more',True),
                              ('a</think>code</think>more',True)]:
            self.assertEqual(local.extract_final_answer(text,thinking)['final_answer'],'')

    def test_truncation_preserves_raw_but_never_usable_text(self):
        for finish in ('length','time','interrupted','error'):
            response=local.response_record(manifest(),1,[row('Reasoning code')],{},finish,1,0)
            self.assertEqual(response['text'],'');self.assertFalse(response['usable_final'])
            self.assertTrue(response['thinking_truncated'])
            self.assertEqual(response['generated_text'],'Reasoning code')
        response=local.response_record(manifest(),1,[row('reason</think>partial code',finish='length')],{},'length',1,0)
        self.assertEqual(response['text'],'');self.assertEqual(response['final_answer'],'partial code')
        self.assertFalse(response['thinking_truncated'])

    def test_finished_response_records_usage_and_final_answer(self):
        response=local.response_record(manifest(),1,[row('think</think>code',finish='stop')],{},'stop',1.5,0)
        self.assertTrue(response['usable_final']);self.assertEqual(response['text'],'code')
        self.assertEqual(response['usage'],{'prompt_tokens':5,'completion_tokens':1,'total_tokens':6})
        self.assertEqual(response['generated_token_ids'],[1])
        self.assertEqual(response['seed'],1)

    def test_partial_decode_recovers_pending_text_and_counts_cannot_exceed_cap(self):
        response=local.response_record(manifest(),1,[row('')],{},'time',1,-9,decode=lambda ids:'a</think>code')
        self.assertEqual(response['generated_text'],'a</think>code')
        self.assertEqual(response['final_answer'],'code');self.assertEqual(response['text'],'')
        with self.assertRaises(ValueError):
            local.response_record(manifest(),1,[row('x',count=2)],{},'time',1,-9)

    def test_existing_output_is_rejected_and_preparation_is_cpu_only(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);prompt=root/'prompt.txt';prompt.write_text('Write a controller.\n')
            model=root/'model.json';model.write_text(json.dumps({'model':'fake','revision':'a'*40}))
            out=root/'screen'
            saved=local.prepare(prompt,out,'on',model_config=model)
            self.assertEqual(saved['max_tokens'],4096);self.assertEqual(saved['max_seconds'],600)
            self.assertEqual((out/'prompt.txt').read_bytes(),prompt.read_bytes())
            self.assertEqual(local.digest(out/'source/program_local.py'),saved['source_sha256'])
            with self.assertRaises(ValueError):local.prepare(prompt,out,'off',model_config=model)
            with self.assertRaises(ValueError):local.prepare(prompt,root/'bad','on',max_seconds=601,model_config=model)
            self.assertFalse((root/'bad').exists())

    def test_lock_refuses_concurrent_screen(self):
        with tempfile.TemporaryDirectory() as root:
            lock=Path(root)/'program-local.lock'
            with local.process_lock(lock):
                with self.assertRaises(RuntimeError):
                    with local.process_lock(lock):pass
            with local.process_lock(lock):pass

    def test_hard_timeout_preserves_completed_cpu_worker_token_records(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root);(out/'source').mkdir()
            fake=out/'source/program_local.py'
            # This child uses only the standard library; it never imports MLX.
            fake.write_text('import json, pathlib, sys, time\n'
                'out=pathlib.Path(sys.argv[2]).parent\n'
                'row={"text":"unfinished thoughts", "token":1, "generation_tokens":1, "prompt_tokens":5, "finish_reason":None}\n'
                '(out/"generation-1-tokens.jsonl").write_text(json.dumps(row)+"\\n")\n'
                'time.sleep(20)\n')
            spec=manifest();local.atomic_json(out/'manifest.json',spec)
            result=local.run_sample(out,spec,1)
            self.assertEqual(result['finish_reason'],'time');self.assertLess(result['elapsed_seconds'],5)
            self.assertEqual(result['completion_tokens'],1);self.assertEqual(result['text'],'')
            self.assertEqual(json.loads((out/'generation-1.json').read_text()),result)

    def test_completed_jsonl_rows_ignore_only_trailing_incomplete_record(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'tokens.jsonl'
            path.write_text(json.dumps(row('a'))+'\n{"incomplete":')
            self.assertEqual(local.complete_token_rows(path),[row('a')])
            path.write_text('{broken}\n')
            with self.assertRaises(json.JSONDecodeError):local.complete_token_rows(path)


if __name__=='__main__':
    unittest.main()
