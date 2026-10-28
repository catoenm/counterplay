"""Small partially observed exchange for competitive RL.

Prices are integer ticks and orders persist until filled or canceled. The two
learners have equal endowments and receive independent, noisy private signals of
an exogenous terminal value. A capitalized market maker supplies *limited*
liquidity; a noise trader supplies independent order flow. Neither background
trader sees a learner's signal or uses a learner's last trade to set value.

``train`` and ``eval`` have identical distributions, with disjoint seeds supplied
by the experiment. ``ood`` doubles terminal-value dispersion, worsens private
signals, halves maker depth and charges one tick per unit per side. This tests
distribution shift rather than merely another realization of the training game.

The public API exposes observations separately from replay-only full snapshots.
Only ``returns`` and terminal ``metrics`` reveal realized financial outcomes.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class _Account:
    cash: int
    inventory: int


@dataclass
class _Order:
    order_id: int
    owner: int
    side: str
    price: int
    quantity: int
    sequence: int


class MarketEnv:
    """Two simultaneous decision makers using a price-time-priority exchange.

    Each round, actions are validated against the same pre-round book. The two
    actions and one exogenous order are then processed in a pre-sampled random
    order. This event ordering is independent of policy choices and reproducible
    across counterfactual policies. Matching uses the resting order's price.
    If the next resting order belongs to the incoming trader, self-trade
    prevention cancels the incoming remainder, preserving price-time priority.
    """

    name = "market"
    INITIAL_CASH = 600
    INITIAL_INVENTORY = 4
    MAKER = 2
    NOISE = 3
    MIN_PRICE = 50
    MAX_PRICE = 150

    def __init__(self, seed: int, regime: str = "train", max_rounds: int = 8):
        if regime not in {"train", "eval", "ood"}:
            raise ValueError("regime must be train, eval, or ood")
        if not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        self.seed = seed
        self.regime = regime
        self.max_rounds = max_rounds
        self.turn = 0
        self.done = False
        self.fee = 1 if regime == "ood" else 0
        self.signal_noise = 6 if regime == "ood" else 3
        self.value_spread = 20 if regime == "ood" else 10
        self.maker_depth = 1 if regime == "ood" else 2
        rng = random.Random(seed)
        self._terminal_value = 100 + rng.randint(-self.value_spread, self.value_spread)
        self._signals = [self._terminal_value + rng.randint(-self.signal_noise, self.signal_noise)
                         for _ in range(2)]
        self._schedule = []
        for round_index in range(max_rounds):
            order = [0, 1, self.NOISE]
            rng.shuffle(order)
            # A weak public signal gradually arrives, independently of trading.
            mid = 100 + round(.4 * round_index / max_rounds * (self._terminal_value - 100))
            mid += rng.randint(-2, 2)
            self._schedule.append({"priority": order, "mid": mid,
                                   "noise_side": rng.choice(["buy", "sell"]),
                                   "noise_quantity": rng.choice([1, 2]),
                                   "noise_offset": rng.choice([-3, 0, 3])})
        self._accounts = {0: _Account(self.INITIAL_CASH, self.INITIAL_INVENTORY),
                          1: _Account(self.INITIAL_CASH, self.INITIAL_INVENTORY),
                          self.MAKER: _Account(1_000_000, 10_000),
                          self.NOISE: _Account(1_000_000, 10_000)}
        self._initial_cash = sum(a.cash for a in self._accounts.values())
        self._initial_inventory = sum(a.inventory for a in self._accounts.values())
        self._orders: list[_Order] = []
        self._next_order_id = 1
        self._trades: list[dict[str, Any]] = []
        self._rounds: list[dict[str, Any]] = []
        self._fees_collected = 0
        self._fees_by_owner = {owner: 0 for owner in self._accounts}
        self._submitted_units = {0: 0, 1: 0}
        self._filled_units = {0: 0, 1: 0}
        self._canceled_units = {0: 0, 1: 0}
        self._refresh_maker()
        self._assert_invariants()

    def active_players(self) -> list[int]:
        return [] if self.done else [0, 1]

    def _check_player(self, player: int) -> None:
        if type(player) is not int or player not in (0, 1):
            raise ValueError("player must be 0 or 1")

    def _reserved(self, owner: int) -> tuple[int, int]:
        cash = sum((o.price + self.fee) * o.quantity for o in self._orders
                   if o.owner == owner and o.side == "buy")
        inventory = sum(o.quantity for o in self._orders if o.owner == owner and o.side == "sell")
        return cash, inventory

    def _available(self, owner: int) -> tuple[int, int]:
        cash, inventory = self._reserved(owner)
        account = self._accounts[owner]
        return account.cash - cash, account.inventory - inventory

    def _cancel(self, owner: int, order_id: int | None = None) -> int:
        canceled = [o for o in self._orders if o.owner == owner
                    and (order_id is None or o.order_id == order_id)]
        ids = {o.order_id for o in canceled}
        self._orders = [o for o in self._orders if o.order_id not in ids]
        units = sum(o.quantity for o in canceled)
        if owner in self._canceled_units:
            self._canceled_units[owner] += units
        return units

    def _submit(self, owner: int, side: str, price: int, quantity: int) -> int:
        """Submit and match one validated limit order; also used by test fixtures."""
        if owner not in self._accounts or side not in {"buy", "sell"}:
            raise ValueError("unknown account or side")
        if (type(price) is not int or not self.MIN_PRICE <= price <= self.MAX_PRICE
                or type(quantity) is not int or quantity < 1):
            raise ValueError("invalid price or quantity")
        cash, inventory = self._available(owner)
        if ((side == "buy" and (price + self.fee) * quantity > cash)
                or (side == "sell" and quantity > inventory)):
            raise ValueError("insufficient unreserved resources")
        order_id = self._next_order_id
        self._next_order_id += 1
        incoming = _Order(order_id, owner, side, price, quantity, order_id)
        self._orders.append(incoming)
        if owner in self._submitted_units:
            self._submitted_units[owner] += quantity
        while incoming.quantity:
            candidates = [o for o in self._orders if o.side != side
                          and o.quantity and (o.price <= price if side == "buy" else o.price >= price)]
            if not candidates:
                break
            resting = min(candidates, key=lambda o: (o.price if side == "buy" else -o.price,
                                                    o.sequence))
            if resting.owner == owner:
                if owner in self._canceled_units:
                    self._canceled_units[owner] += incoming.quantity
                incoming.quantity = 0
                self._orders = [o for o in self._orders if o.quantity]
                break
            units = min(incoming.quantity, resting.quantity)
            trade_price = resting.price
            buyer, seller = (owner, resting.owner) if side == "buy" else (resting.owner, owner)
            self._accounts[buyer].cash -= (trade_price + self.fee) * units
            self._accounts[buyer].inventory += units
            self._accounts[seller].cash += (trade_price - self.fee) * units
            self._accounts[seller].inventory -= units
            self._fees_collected += 2 * self.fee * units
            for participant in (buyer, seller):
                self._fees_by_owner[participant] += self.fee * units
                if participant in self._filled_units:
                    self._filled_units[participant] += units
            self._trades.append({"round": self.turn, "price": trade_price, "quantity": units,
                                 "buyer": buyer, "seller": seller,
                                 "resting_order_id": resting.order_id,
                                 "incoming_order_id": incoming.order_id})
            incoming.quantity -= units
            resting.quantity -= units
            self._orders = [o for o in self._orders if o.quantity]
        self._assert_invariants()
        return order_id

    def _refresh_maker(self) -> None:
        self._cancel(self.MAKER)
        self._cancel(self.NOISE)
        mid = self._schedule[self.turn]["mid"]
        self._submit(self.MAKER, "buy", mid - 2, self.maker_depth)
        self._submit(self.MAKER, "sell", mid + 2, self.maker_depth)

    def _book(self, side: str) -> list[dict[str, int]]:
        levels: dict[int, int] = {}
        for o in self._orders:
            if o.side == side:
                levels[o.price] = levels.get(o.price, 0) + o.quantity
        return [{"price": price, "quantity": levels[price]}
                for price in sorted(levels, reverse=side == "buy")]

    def observation(self, player: int) -> dict[str, Any]:
        self._check_player(player)
        cash_reserved, inventory_reserved = self._reserved(player)
        account = self._accounts[player]
        return {"game": self.name, "player": player, "round": self.turn,
                "rounds_remaining": self.max_rounds - self.turn,
                "rules": {"objective": "Maximize terminal profit relative to the other trader.",
                          "settlement": "Inventory settles at a common hidden exogenous value; last trade cannot change it.",
                          "signal": f"Your fixed private signal equals terminal value plus uniform integer noise +/-{self.signal_noise}.",
                          "value_prior": [100 - self.value_spread, 100 + self.value_spread],
                          "execution": "Persistent limit orders, price-time priority, resting price. Simultaneous choices have random event priority. No borrowing or self-trades.",
                          "fee_per_unit_per_side": self.fee},
                "private_signal": self._signals[player],
                "account": {"cash": account.cash, "inventory": account.inventory,
                            "reserved_cash": cash_reserved, "reserved_inventory": inventory_reserved},
                "book": {"bids": self._book("buy"), "asks": self._book("sell")},
                "own_orders": [{"id": o.order_id, "side": o.side, "price": o.price, "quantity": o.quantity}
                               for o in self._orders if o.owner == player],
                "recent_trades": [{k: t[k] for k in ("round", "price", "quantity")}
                                  for t in self._trades[-4:]]}

    def legal_actions(self, player: int) -> list[dict[str, Any]]:
        self._check_player(player)
        if self.done:
            return []
        result: list[dict[str, Any]] = [{"type": "hold"}]
        if any(o.owner == player for o in self._orders):
            result.append({"type": "cancel"})
        mid = self._schedule[self.turn]["mid"]
        bids, asks = self._book("buy"), self._book("sell")
        # The menu supports joining quotes, placing an inside quote, and crossing.
        prices = sorted({bids[0]["price"] if bids else mid - 2, mid,
                         asks[0]["price"] if asks else mid + 2})
        available_cash, available_inventory = self._available(player)
        for side in ("buy", "sell"):
            for price in prices:
                for quantity in (1, 2):
                    if (side == "buy" and (price + self.fee) * quantity <= available_cash
                            or side == "sell" and quantity <= available_inventory):
                        result.append({"type": "limit", "side": side,
                                       "price": price, "quantity": quantity})
        return result

    def step(self, actions: dict[int, dict[str, Any]]) -> None:
        if self.done:
            raise ValueError("episode is already terminal")
        if set(actions) != {0, 1} or any(type(k) is not int for k in actions):
            raise ValueError("both simultaneous actions are required")
        for player in (0, 1):
            action = actions[player]
            if (not isinstance(action, dict) or action not in self.legal_actions(player)
                    or any(type(action[k]) is not int for k in ("price", "quantity") if k in action)):
                raise ValueError(f"illegal action for player {player}")
        event = self._schedule[self.turn]
        before = len(self._trades)
        for player in event["priority"]:
            if player == self.NOISE:
                self._submit(player, event["noise_side"], event["mid"] + event["noise_offset"],
                             event["noise_quantity"])
            else:
                action = actions[player]
                if action["type"] == "cancel":
                    self._cancel(player)
                elif action["type"] == "limit":
                    self._submit(player, action["side"], action["price"], action["quantity"])
        self._rounds.append({"round": self.turn, "actions": copy.deepcopy(actions),
                             "event_priority": list(event["priority"]),
                             "trades": copy.deepcopy(self._trades[before:])})
        self.turn += 1
        self.done = self.turn >= self.max_rounds
        if self.done:
            for owner in self._accounts:
                self._cancel(owner)
        else:
            self._refresh_maker()
        self._assert_invariants()

    def _pnl(self, player: int) -> int:
        account = self._accounts[player]
        return (account.cash - self.INITIAL_CASH
                + (account.inventory - self.INITIAL_INVENTORY) * self._terminal_value)

    def returns(self) -> list[float]:
        if not self.done:
            raise ValueError("returns are only available after settlement")
        relative = math.tanh((self._pnl(0) - self._pnl(1)) / (8 * self.max_rounds))
        return [relative, -relative]

    def metrics(self) -> dict[str, Any]:
        return {"game": self.name, "regime": self.regime, "rounds": self.turn,
                "terminal": self.done,
                "terminal_value": self._terminal_value if self.done else None,
                "absolute_pnl": [self._pnl(p) for p in (0, 1)] if self.done else None,
                "excess_wealth_return": [self._pnl(p) / 1000 for p in (0, 1)] if self.done else None,
                "returns": self.returns() if self.done else None,
                "reward_formula": "tanh((pnl_0-pnl_1)/(8*max_rounds)); player 1 receives its negative",
                "cash": [self._accounts[p].cash for p in (0, 1)],
                "inventory": [self._accounts[p].inventory for p in (0, 1)],
                "fees": [self._fees_by_owner[p] for p in (0, 1)],
                "filled_units": [self._filled_units[p] for p in (0, 1)],
                "submitted_units": [self._submitted_units[p] for p in (0, 1)],
                "trade_count": len(self._trades), "exchange_fees": self._fees_collected}

    def snapshot(self) -> dict[str, Any]:
        """Omniscient replay state. Never provide this object to a policy."""
        return {"game": self.name, "seed": self.seed, "regime": self.regime,
                "turn": self.turn, "done": self.done, "max_rounds": self.max_rounds,
                "terminal_value": self._terminal_value, "private_signals": list(self._signals),
                "accounts": {p: asdict(a) for p, a in self._accounts.items()},
                "orders": [asdict(o) for o in self._orders],
                "trades": copy.deepcopy(self._trades), "rounds": copy.deepcopy(self._rounds),
                "schedule": copy.deepcopy(self._schedule), "metrics": self.metrics()}

    def heuristic_action(self, player: int) -> dict[str, Any]:
        """A private-signal value trader, using only its own policy-visible state."""
        obs = self.observation(player)
        choices = self.legal_actions(player)
        if not choices:
            raise ValueError("episode is already terminal")
        fee = obs["rules"]["fee_per_unit_per_side"]
        # Clip the noisy signal to the public prior's support.
        lower, upper = obs["rules"]["value_prior"]
        estimate = max(lower, min(upper, obs["private_signal"]))
        own_orders = obs["own_orders"]
        if own_orders:
            stale = any((estimate - o["price"] if o["side"] == "buy" else o["price"] - estimate)
                        <= fee for o in own_orders)
            if stale:
                return {"type": "cancel"}
        best, best_score = choices[0], 0.0
        for action in choices:
            if action["type"] != "limit":
                continue
            side, price, quantity = action["side"], action["price"], action["quantity"]
            edge = (estimate - price if side == "buy" else price - estimate) - fee
            opposite = obs["book"]["asks" if side == "buy" else "bids"]
            executable = sum(level["quantity"] for level in opposite
                             if (level["price"] <= price if side == "buy" else level["price"] >= price))
            # Prefer executable positive-edge orders; allow a smaller passive bid.
            score = edge * min(quantity, executable) if executable else .1 * edge * quantity
            if score > best_score:
                best, best_score = action, score
        return dict(best)

    def _assert_invariants(self) -> None:
        assert sum(a.cash for a in self._accounts.values()) + self._fees_collected == self._initial_cash
        assert sum(a.inventory for a in self._accounts.values()) == self._initial_inventory
        assert len({o.order_id for o in self._orders}) == len(self._orders)
        for owner, account in self._accounts.items():
            assert account.cash >= 0 and account.inventory >= 0
            available_cash, available_inventory = self._available(owner)
            assert available_cash >= 0 and available_inventory >= 0
        assert all(o.quantity > 0 for o in self._orders)
        assert all(t["buyer"] != t["seller"] for t in self._trades)
