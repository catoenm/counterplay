"""Reverify saved arena episodes and embed their real states in the replay.

Run after any experiment stage completes; no model inference or network access.
An empty run directory produces an explicitly empty, awaiting-records display.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfplay.arena.chess_game import ChessEnv
from selfplay.arena.market import MarketEnv
from selfplay.arena.maze import MazeEnv


DEFAULT_FRAGMENT = Path(
    "/Users/mitchellcatoen/.codex/visualizations/2026/10/20/"
    "01a15b4f-05a3-7053-9034-4d1bc9f8d726/selfplay-arena.html")


def normalized(value):
    return json.loads(json.dumps(value))


def compact_state(game, state, previous_trades=0):
    if game == "maze":
        keys = ("size", "positions", "walls", "terrain", "goal", "blocks_remaining", "done")
        result = {key: state[key] for key in keys}
        result["distances"] = state["metrics"]["shortest_distances"]
        result["cancelled"] = state["history"][-1]["cancelled_blocks"] if state["history"] else []
        return result
    if game == "chess":
        return {"fen": state["fen"], "done": state["done"],
                "in_check": state["in_check"], "material": state["material"],
                "side_to_move": state["side_to_move"]}
    return {"accounts": {str(p): state["accounts"][str(p)] for p in (0, 1)},
            "orders": state["orders"], "signals": state["private_signals"],
            "trades": state["trades"][previous_trades:],
            "last_trade": state["trades"][-1] if state["trades"] else None,
            "settlement": state["terminal_value"] if state["done"] else None,
            "pnl": state["metrics"]["absolute_pnl"], "done": state["done"]}


def verify_and_compact(row, source):
    game, initial = row["environment"], row["initial_state"]
    classes = {"market": MarketEnv, "maze": MazeEnv, "chess": ChessEnv}
    horizon_key = {"market": "max_rounds", "maze": "round_limit", "chess": "ply_limit"}[game]
    env = classes[game](row["seed"], row["regime"], initial[horizon_key])
    if normalized(env.snapshot()) != initial:
        raise ValueError(f"Initial state no longer replays: {row['episode_id']}")
    states, actions = [compact_state(game, initial)], []
    previous_trades = len(initial.get("trades", []))
    for event in row["events"]:
        if event["turn"] != env.turn:
            raise ValueError(f"Out-of-order turn: {row['episode_id']}")
        chosen = {int(player): action for player, action in event["actions"].items()}
        env.step(chosen)
        state = normalized(env.snapshot())
        if state != event["state"]:
            raise ValueError(f"Recorded state does not reverify: {row['episode_id']} turn {event['turn']}")
        states.append(compact_state(game, state, previous_trades))
        previous_trades = len(state.get("trades", []))
        actions.append(event["actions"])
    if not env.done or env.returns() != row["returns"] or normalized(env.metrics()) != row["metrics"]:
        raise ValueError(f"Terminal result does not reverify: {row['episode_id']}")
    checkpoint = row.get("candidate", row["policies"][0])
    return {"id": row["episode_id"], "environment": game, "seed": row["seed"],
            "checkpoint": checkpoint, "phase": "training" if source.startswith("train-") else
                          "league" if source.startswith("league-") else "evaluation",
            "policies": row["policies"], "seat": row.get("candidate_seat"),
            "returns": row["returns"], "reason": row["metrics"].get("finish_reason", "settlement"),
            "states": states, "actions": actions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/arena-v1"))
    ap.add_argument("--fragment", type=Path, default=DEFAULT_FRAGMENT)
    args = ap.parse_args()
    episodes, sources = [], {}
    manifest_path = args.run / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes() if manifest_path.exists() else None
    manifest = json.loads(manifest_bytes) if manifest_bytes is not None else {}
    arithmetic = manifest.get('policy_arithmetic')
    if arithmetic is not None:
        arithmetic_bytes = json.dumps(arithmetic, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if hashlib.sha256(arithmetic_bytes).hexdigest() != manifest.get('policy_arithmetic_sha256'):
            raise ValueError('Manifest arithmetic profile hash differs')
    for path in sorted(args.run.glob("*.jsonl")):
        if not path.name.startswith(("eval-", "train-rollouts-", "league-")):
            continue
        content = path.read_bytes()
        sources[path.name] = hashlib.sha256(content).hexdigest()
        lines = content.splitlines(keepends=True)
        # The writer flushes one complete episode per newline. A currently
        # incomplete trailing write is excluded; never manufacture a completion.
        for raw in lines:
            if not raw.endswith(b"\n"):
                continue
            episodes.append(verify_and_compact(json.loads(raw), path.stem))
    ids = [episode["id"] for episode in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode identity")
    episodes.sort(key=lambda e: (e["environment"], e["checkpoint"], e["phase"], e["id"]))
    encoded = json.dumps({"run": args.run.name, "verified": True, "episodes": episodes,
                          "policy_arithmetic": arithmetic,
                          "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None,
                          "sources_sha256": sources}, separators=(",", ":")).replace("<", "\\u003c")
    text, replaced = re.subn(r'(<script id="sa-data" type="application/json">).*?(</script>)',
                             lambda match: match[1] + encoded + match[2],
                             args.fragment.read_text(), flags=re.S)
    if replaced != 1:
        raise ValueError("Expected one replay data block")
    if len(text.encode()) > 1_000_000:
        raise ValueError("Replay exceeds 1 MB; reduce embedded fields before publishing")
    args.fragment.write_text(text)
    print(json.dumps({"episodes": len(episodes), "steps": sum(len(e["actions"]) for e in episodes),
                      "bytes": len(text.encode()), "fragment": str(args.fragment)}))


if __name__ == "__main__":
    main()
