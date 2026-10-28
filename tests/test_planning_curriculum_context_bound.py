"""CPU tokenizer/schema checks; no model loading or sampled policy actions."""
from pathlib import Path
import unittest

from transformers import AutoTokenizer

from selfplay.arena_mixed import planning, cuda_policy as renderer
from selfplay.planning_curriculum.admission import build_context_bound
from selfplay.planning_curriculum import envs


class ContextBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parents[1]/"runs/arena-mixed-preflight-v1/tokenizer"
        cls.bound = build_context_bound(cls.directory)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.directory, local_files_only=True, trust_remote_code=False)

    def test_native_byte_bound_is_not_observed_example_token_count(self):
        bound = self.bound
        self.assertEqual(bound["domain"]["tool_subsets_checked"], 35)
        self.assertEqual(bound["domain"]["maximum_history_rows"], 11)
        self.assertFalse(bound["domain"]["synthetic_envelope_is_an_episode"])
        self.assertGreater(bound["maximum_input_tokens"], bound["envelope_actual_tokens_descriptive_only"])
        self.assertLessEqual(bound["maximum_input_tokens"], 4096)
        self.assertNotIn("review_sha256", bound)  # A builder cannot approve its own proof.
        self.assertEqual(len(bound["proof_sha256"]), 64)

    def test_actual_bounded_public_trajectories_and_menu_orders_fit(self):
        # Fixed CPU transition examples supplement the finite schema proof;
        # they are not model episodes or empirical replacements for the bound.
        count = 0
        for seed in range(18):
            env = envs.PlanningEnv(seed, depth=1+seed % 6)
            while not env.done:
                obs = env.observation(0)
                for menu in (env.legal_actions(0), list(reversed(env.legal_actions(0)))):
                    prompt = renderer.action_prompt(obs, menu, list("ABCD"))
                    rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    tokens = self.tokenizer.encode(rendered, add_special_tokens=False)
                    self.assertLessEqual(len(tokens), len(rendered.encode()))
                    self.assertLessEqual(len(rendered.encode()), self.bound["maximum_input_tokens"])
                    self.assertNotIn("witness", prompt)
                    count += 1
                env.step({0: env.legal_actions(0)[env.turn % 4]})
        self.assertGreater(count, 100)


if __name__ == "__main__": unittest.main()
