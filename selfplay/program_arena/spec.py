"""Model-facing language and game rules, without a reference solution."""


def model_spec():
    return '''Write one executable robot controller as a plain JSON object, with no markdown or explanation.
Your controller is run once per round and must emit one action. A rival controller runs simultaneously.

GAME: A 7x7 grid (x,y from 0 to 6), 48 rounds. You start at your home, either (0,3) or (6,3).
Six finite resource sites are symmetric initially but shared: either robot can harvest any site.
Each site has x,y,units,value; a harvested unit is worth value points. Moving one cardinal cell costs
one energy. Harvesting a unit on your current cell costs one energy; cargo capacity is three units.
Rest restores three energy, up to ten. Deposit at your own home banks all cargo for points.
Only banked points count at the end. Cargo left in transit does not score. Robots can share cells.
If both harvest the last unit on the same turn, seeded random priority decides who gets it.
Maximize your banked points relative to the opponent. No other rewards, hazards, walls or actions.
Failed moves/harvests/deposits consume the round. Signals contain no future random priority.

PROGRAM: {"version":1,"body":[STATEMENT,...]}. All blocks must be nonempty.
Statements (exact keys):
 {"set":"v_name","value":EXPRESSION} assigns a local; m0 through m15 assign persistent integer memory.
 {"if":EXPRESSION,"then":[STATEMENT,...],"else":[STATEMENT,...]} branches on nonzero.
 {"repeat":LITERAL_INTEGER_0_TO_16,"index":"v_i","body":[STATEMENT,...]} bounded loop; index starts at 0.
 {"emit":{"action":"move","dx":EXPRESSION,"dy":EXPRESSION}} terminates this turn, with abs(dx)+abs(dy)=1.
 {"emit":{"action":"harvest"}} or {"emit":{"action":"deposit"}} or {"emit":{"action":"rest"}} terminates this turn.
Do not use Python or extra keys. Sensors cannot be assigned. Local names match v_[a-z][a-z0-9_]{0,19}.
Locals start at zero each turn. Persistent m0..m15 start at zero once per game and survive successful turns.

EXPRESSION forms:
 an integer between -1000000 and 1000000 (no floats, booleans or strings);
 {"var":"NAME"};
 {"op":"OPERATOR","args":[EXPRESSION,...]};
 {"site":{"index":EXPRESSION,"field":"FIELD"}}, where FIELD is x, y, units or value.
An out-of-range site index reads zero. Binary operators: + - * // % min max < <= > >= == != and or.
Unary operators: abs neg not. Comparisons/logic return 0 or 1. Arithmetic saturates at +/-1000000.
Both arguments of and/or are evaluated. Division/modulo by zero faults. There is no exponentiation.

Read-only sensor NAMEs: player, turn, horizon, grid_size, self_x, self_y, home_x, home_y, energy,
max_energy, carrying, capacity, score, opponent_x, opponent_y, opponent_energy, opponent_carrying,
opponent_score, site_count. All are integers. carrying is a unit count; site_count is six.
No list indexing syntax: inspect sites using the site expression above. No function definitions or calls.

LIMITS: source <=16384 UTF-8 bytes, <=512 AST nodes, nesting <=20, <=32 local names,
<=2048 interpreter operations per turn, 16 persistent memory slots, repeat count <=16.
Every statement, expression and loop iteration costs an operation. A runtime fault or no emitted action
causes a rest action and discards that turn's memory writes; it is recorded as a program fault.
Your output must be a complete controller, not a list of actions for a known seed.
Minimal syntax-only example (not a useful strategy): {"version":1,"body":[{"emit":{"action":"rest"}}]}'''
