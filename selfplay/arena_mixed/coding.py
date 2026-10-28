"""Single-player bounded AST repair with public feedback and sealed test cases.

Programs are parsed into a small fixed AST subset and interpreted. No Python
source is executed or compiled to bytecode. This measures edit decisions on
parametric repair families, not unrestricted code authorship.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import random


CONSTANTS = tuple(range(-4, 5))
BINOPS = {'Add':ast.Add, 'Sub':ast.Sub, 'Mult':ast.Mult}
COMPARES = {'Lt':ast.Lt, 'LtE':ast.LtE, 'Gt':ast.Gt, 'GtE':ast.GtE, 'Eq':ast.Eq, 'NotEq':ast.NotEq}
MAX_NODES, MAX_INPUT, MAX_OPERATIONS, MAX_INTEGER = 128, 8, 2048, 1_000_000


class ProgramError(ValueError):
    pass


def parse_program(source):
    if not isinstance(source, str) or len(source.encode()) > 4096:
        raise ProgramError('Program byte bound exceeded')
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as error:
        raise ProgramError('Invalid program syntax') from error
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ProgramError('Exactly one bounded function required')
    fn = tree.body[0]
    if (fn.name != 'solve' or fn.decorator_list or fn.returns or fn.type_comment
            or len(fn.args.args) != 1 or fn.args.args[0].arg != 'xs'
            or fn.args.args[0].annotation or fn.args.posonlyargs or fn.args.kwonlyargs
            or fn.args.defaults or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg):
        raise ProgramError('Expected def solve(xs)')
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.For,
               ast.If, ast.Return, ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp,
               ast.Compare, ast.UnaryOp, ast.USub, *BINOPS.values(), *COMPARES.values())
    for node in nodes:
        if not isinstance(node, allowed):
            raise ProgramError('Unsupported AST node: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in ('xs', 'item', 'result'):
            raise ProgramError('Unknown variable')
        if isinstance(node, ast.Constant) and (type(node.value) is not int or not -64 <= node.value <= 64):
            raise ProgramError('Only bounded integer constants')
        if isinstance(node, ast.Assign) and (len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)
                                             or node.targets[0].id != 'result' or node.type_comment):
            raise ProgramError('Only scalar result may be assigned')
        if isinstance(node, ast.For):
            if (not isinstance(node.target, ast.Name) or node.target.id != 'item'
                    or not isinstance(node.iter, ast.Name) or node.iter.id != 'xs'
                    or node.orelse or node.type_comment
                    or any(isinstance(n, ast.For) for statement in node.body for n in ast.walk(statement))):
                raise ProgramError('Only one nonnested for item in xs loop')
        if isinstance(node, ast.Compare) and (len(node.ops) != 1 or len(node.comparators) != 1):
            raise ProgramError('Only single numeric comparisons')
    if sum(isinstance(n, ast.For) for n in nodes) != 1:
        raise ProgramError('Exactly one bounded list loop required')
    if not fn.body or not isinstance(fn.body[-1], ast.Return):
        raise ProgramError('Final scalar return required')
    return tree


def run_program(source, inputs):
    tree = parse_program(source)
    if (not isinstance(inputs, list) or len(inputs) > MAX_INPUT
            or any(type(x) is not int or abs(x) > 64 for x in inputs)):
        raise ProgramError('Inputs exceed fixed integer-list bounds')
    state, operations = {'xs':list(inputs)}, 0

    def tick():
        nonlocal operations
        operations += 1
        if operations > MAX_OPERATIONS:
            raise ProgramError('Interpreter operation limit exceeded')

    def expr(node):
        tick()
        if isinstance(node, ast.Constant): value = node.value
        elif isinstance(node, ast.Name):
            if node.id not in state or type(state[node.id]) is not int:
                raise ProgramError('Uninitialized or nonscalar expression')
            value = state[node.id]
        elif isinstance(node, ast.UnaryOp): value = -expr(node.operand)
        elif isinstance(node, ast.BinOp):
            left, right = expr(node.left), expr(node.right)
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            else: value = left * right
        elif isinstance(node, ast.Compare):
            left, right = expr(node.left), expr(node.comparators[0]); op = node.ops[0]
            if isinstance(op, ast.Lt): value = int(left < right)
            elif isinstance(op, ast.LtE): value = int(left <= right)
            elif isinstance(op, ast.Gt): value = int(left > right)
            elif isinstance(op, ast.GtE): value = int(left >= right)
            elif isinstance(op, ast.Eq): value = int(left == right)
            else: value = int(left != right)
        else: raise ProgramError('Expression outside fixed interpreter')
        if type(value) is not int or abs(value) > MAX_INTEGER:
            raise ProgramError('Integer bound exceeded')
        return value

    def block(statements):
        for node in statements:
            tick()
            if isinstance(node, ast.Assign): state['result'] = expr(node.value)
            elif isinstance(node, ast.If):
                result = block(node.body if expr(node.test) else node.orelse)
                if result is not None:return result
            elif isinstance(node, ast.For):
                for item in state['xs']:
                    state['item'] = item
                    result = block(node.body)
                    if result is not None:return result
            elif isinstance(node, ast.Return): return expr(node.value)
            else: raise ProgramError('Statement outside fixed interpreter')
        return None
    result = block(tree.body[0].body)
    if type(result) is not int:raise ProgramError('No integer result')
    return result


def edit_sites(source):
    """Stable walk slots; menus depend only on the visible current AST."""
    sites = []
    for index, node in enumerate(ast.walk(parse_program(source))):
        if isinstance(node, ast.Constant): kind, current, options = 'constant', node.value, CONSTANTS
        elif isinstance(node, ast.BinOp): kind, current, options = 'operator', type(node.op).__name__, tuple(BINOPS)
        elif isinstance(node, ast.Compare): kind, current, options = 'comparison', type(node.ops[0]).__name__, tuple(COMPARES)
        else: continue
        sites.append({'slot':index, 'line':node.lineno, 'kind':kind, 'current':current,
                      'options':list(options)})
    return sites


def apply_edit(source, action):
    if (not isinstance(action, dict) or set(action) != {'type','slot','value'}
            or action['type'] != 'edit' or type(action['slot']) is not int):
        raise ProgramError('Invalid AST edit')
    sites = {s['slot']:s for s in edit_sites(source)}
    site = sites.get(action['slot'])
    if (site is None or action['value'] == site['current'] or action['value'] not in site['options']
            or (site['kind']=='constant' and type(action['value']) is not int)):
        raise ProgramError('Edit is not in the visible legal menu')
    tree = parse_program(source); node = list(ast.walk(tree))[action['slot']]
    if site['kind'] == 'constant': node.value = action['value']
    elif site['kind'] == 'operator': node.op = BINOPS[action['value']]()
    else: node.ops = [COMPARES[action['value']]()]
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    parse_program(result)
    return result


def reference(family, parameter, xs):
    """Trusted task semantics, independent of the candidate interpreter."""
    if family == 'sum_shift':return sum(x + parameter for x in xs)
    if family == 'count_at_least':return sum(x >= parameter for x in xs)
    if family == 'maximum_with_floor':return max([parameter] + xs)
    raise ValueError('Unknown task family')


def task_source(family, parameter):
    if family == 'sum_shift':
        body = f'result = 0\nfor item in xs:\n    result = result + item + ({parameter})\nreturn result'
    elif family == 'count_at_least':
        body = f'result = 0\nfor item in xs:\n    if item >= ({parameter}):\n        result = result + 1\nreturn result'
    else:
        body = f'result = ({parameter})\nfor item in xs:\n    if item > result:\n        result = item\nreturn result'
    return ast.unparse(parse_program('def solve(xs):\n' + '\n'.join('    '+line for line in body.splitlines())))+'\n'


def cases_for(family, parameter, inputs):
    return [{'input':list(xs), 'expected':reference(family, parameter, xs)} for xs in inputs]


def evaluate(source, cases):
    rows=[]
    for case in cases:
        try:
            actual=run_program(source,case['input']);error=None
        except ProgramError as e:
            actual=None;error=str(e)
        rows.append({**deepcopy(case),'actual':actual,'passed':error is None and actual==case['expected'],'error':error})
    return rows


class CodingEnv:
    name='coding'
    role_count=1

    def __init__(self,seed,regime='train',max_rounds=12):
        if type(seed) is not int or seed<0 or regime not in ('train','eval','ood'):
            raise ValueError('Invalid coding seed/regime')
        if type(max_rounds) is not int or not 3<=max_rounds<=16:
            raise ValueError('Coding horizon must be in3..16')
        self.seed,self.regime,self.max_rounds=seed,regime,max_rounds
        rng=random.Random(seed);self.family=('sum_shift','count_at_least','maximum_with_floor')[seed%3]
        self.parameter=rng.randint(-3,3)
        self._solution=task_source(self.family,self.parameter)
        self.specification={'family':self.family,'parameter':self.parameter,'input':'Integer list xs of length0..8.',
            'objective':{'sum_shift':f'Return the sum of (item + {self.parameter}) for every item; empty list returns0.',
                         'count_at_least':f'Return the number of items greater than or equal to {self.parameter}.',
                         'maximum_with_floor':f'Return the largest value among {self.parameter} and all items.'}[self.family]}
        public=[[],[0],[-1,2],[3,-2,1]]
        hidden=[[-8,-7], [self.parameter,self.parameter], [self.parameter-1,self.parameter+1]]
        used={tuple(x) for x in public};hidden=[x for x in hidden if tuple(x) not in used]
        used.update(tuple(x) for x in hidden)
        bound=12 if regime=='ood' else 8
        while len(hidden)<16:
            xs=[rng.randint(-bound,bound) for _ in range(rng.randint(1,8))]
            if tuple(xs) not in used:used.add(tuple(xs));hidden.append(xs)
        self.public_tests=cases_for(self.family,self.parameter,public)
        self._hidden_tests=cases_for(self.family,self.parameter,hidden)
        edits=[{'type':'edit','slot':s['slot'],'value':v} for s in edit_sites(self._solution)
               for v in s['options'] if v!=s['current']]
        rng.shuffle(edits)
        # Initial bugs are selected only using declared public examples. There
        # is a one-edit exact solution, but neither its source nor edit is shown.
        for edit in edits:
            buggy=apply_edit(self._solution,edit)
            if not all(row['passed'] for row in evaluate(buggy,self.public_tests)):
                self.initial_program=buggy;break
        else:raise RuntimeError('Unable to construct a nontrivial visible bug')
        self.program=self.initial_program;self.turn=0;self.done=False;self.finish_reason=None
        self.history=[];self.public_feedback=None;self._final_rows=None;self._result=None

    @staticmethod
    def _player(player):
        if type(player) is not int or player!=0:raise ValueError('Coding has exactly one role:0')

    def active_players(self):return [] if self.done else [0]

    def legal_actions(self,player):
        self._player(player)
        if self.done:return []
        actions=[{'type':'edit','slot':s['slot'],'value':v} for s in edit_sites(self.program)
                 for v in s['options'] if v!=s['current']]
        actions += [{'type':'run_public_tests'},{'type':'submit'}]
        if len(actions)>256:raise ValueError('Bounded repair menu exceeded256')
        return actions

    def observation(self,player):
        self._player(player)
        return {'environment':self.name,'player':0,'role_count':1,'turn':self.turn,'round_limit':self.max_rounds,
            'specification':deepcopy(self.specification),'program':self.program,'edit_sites':edit_sites(self.program),
            'public_tests':deepcopy(self.public_tests),'public_feedback':deepcopy(self.public_feedback),
            'history':deepcopy(self.history),'done':self.done,
            'rules':'Single-player bounded Python AST repair. Each edit replaces one visible constant, arithmetic '
                    'operator or comparison at a listed slot; slots are refreshed after edits. run_public_tests '
                    'reports only public cases. submit ends the episode and scores hidden tests. The turn cap '
                    'also submits the current program. Every action costs one turn. Reward is hidden pass fraction '
                    'in[0,1]. Programs run only in the bounded interpreter. There is no opponent role.'}

    def step(self,actions):
        if self.done:raise RuntimeError('Episode already terminal')
        if (not isinstance(actions,dict) or list(actions)!=[0] or type(next(iter(actions))) is not int
                or actions[0] not in self.legal_actions(0)):
            raise ValueError('Submit exactly one legal role0 action')
        action=deepcopy(actions[0]);kind=action['type']
        if kind=='edit':self.program=apply_edit(self.program,action);self.public_feedback=None
        elif kind=='run_public_tests':self.public_feedback=evaluate(self.program,self.public_tests)
        self.turn+=1
        self.history.append({'turn':self.turn,'action':action,'program_sha256':hashlib.sha256(self.program.encode()).hexdigest(),
                             'public_feedback':deepcopy(self.public_feedback) if kind=='run_public_tests' else None})
        if kind=='submit' or self.turn>=self.max_rounds:
            self.done=True;self.finish_reason='submitted' if kind=='submit' else 'repair_step_cap'
            self._final_rows=evaluate(self.program,self._hidden_tests)
            self._result=sum(r['passed'] for r in self._final_rows)/len(self._final_rows)

    def returns(self):
        if not self.done:raise RuntimeError('Returns only exist at termination')
        return [float(self._result),0.0]

    def metrics(self):
        return {'turns':self.turn,'finish_reason':self.finish_reason,'role_count':1,
            'hidden_pass_fraction':self._result,'hidden_all_passed':self._result==1 if self.done else None,
            'hidden_runtime_faults':sum(r['error'] is not None for r in self._final_rows) if self.done else None,
            'edits':sum(h['action']['type']=='edit' for h in self.history),
            'public_test_runs':sum(h['action']['type']=='run_public_tests' for h in self.history),
            'returns':self.returns() if self.done else None}

    def fingerprint_payload(self):
        return {'environment':self.name,'task_family':self.family,'parameter':self.parameter,
                'specification':deepcopy(self.specification),'initial_program':self.initial_program,
                'public_tests':deepcopy(self.public_tests)}

    def snapshot(self):
        return {**self.observation(0),'seed':self.seed,'regime':self.regime,'metrics':self.metrics(),
            'scenario_identity':self.fingerprint_payload(),
            'audit_only':{'hidden_tests':deepcopy(self._hidden_tests),'reference_solution':self._solution,
                          'final_hidden_results':deepcopy(self._final_rows)}}

    def heuristic_action(self,player):
        """One-edit public-test search benchmark, never a hidden-test oracle."""
        self._player(player)
        if self.done:raise RuntimeError('No active action')
        current=sum(r['passed'] for r in evaluate(self.program,self.public_tests))
        if current==len(self.public_tests):return {'type':'submit'}
        best=(current,None)
        for action in self.legal_actions(0):
            if action['type']!='edit':continue
            passed=sum(r['passed'] for r in evaluate(apply_edit(self.program,action),self.public_tests))
            if passed>best[0]:best=(passed,action)
        return deepcopy(best[1]) if best[1] is not None else {'type':'submit'}

    def assert_private_boundary(self):
        before=(self.observation(0),self.legal_actions(0));other=deepcopy(self)
        other._hidden_tests=[{'input':[64],'expected':-999}]
        other._solution='withheld';other._final_rows=None
        if before!=(other.observation(0),other.legal_actions(0)):
            raise ValueError('Hidden coding cases or solution leaked')
        return {'checked':True,'roles':[0],'private_fields':['hidden_tests','reference_solution']}
