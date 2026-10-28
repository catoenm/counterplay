"""Hand-authored programs for compiler tests and fixed benchmark opponents.

These examples are not model-generated and should not be copied into a model's
generation prompt when measuring its ability to develop an algorithm unaided.
"""


def var(name):
    return {"var": name}


def op(name, *args):
    return {"op": name, "args": list(args)}


def site(field):
    return {"site": {"index": var("v_i"), "field": field}}


def assign(name, value):
    return {"set": name, "value": value}


def emit(name, **kwargs):
    return {"emit": {"action": name, **kwargs}}


def branch(condition, then, otherwise=None):
    return {"if": condition, "then": then, "else": otherwise or [assign("v_noop", 0)]}


def idle_program():
    return {"version": 1, "body": [emit("rest")]}


def collector_program(distance_weight=100, value_weight=1, opponent_weight=0):
    """An actual resource-scanning algorithm parameterized by decision weights."""
    same_position = op("and", op("==", var("self_x"), var("v_tx")), op("==", var("self_y"), var("v_ty")))
    at_home = op("and", op("==", var("self_x"), var("home_x")), op("==", var("self_y"), var("home_y")))
    distance = op("+", op("abs", op("-", site("x"), var("self_x"))), op("abs", op("-", site("y"), var("self_y"))))
    opponent_distance = op("+", op("abs", op("-", site("x"), var("opponent_x"))), op("abs", op("-", site("y"), var("opponent_y"))))
    cost = op("-", op("-", op("*", var("v_distance"), distance_weight), op("*", site("value"), value_weight)),
              op("*", opponent_distance, opponent_weight))
    choose_site = [assign("v_distance", distance), assign("v_cost", cost),
                   branch(op("<", var("v_cost"), var("v_best")),
                          [assign("v_best", var("v_cost")), assign("v_tx", site("x")),
                           assign("v_ty", site("y")), assign("v_target", var("v_i"))])]
    return_home = [assign("v_tx", var("home_x")), assign("v_ty", var("home_y")), assign("v_target", -1)]
    home_distance = op("+", op("abs", op("-", var("home_x"), var("self_x"))),
                       op("abs", op("-", var("home_y"), var("self_y"))))
    return {"version": 1, "body": [
        assign("m0", op("+", var("m0"), 1)),
        branch(op("==", var("energy"), 0), [emit("rest")]),
        branch(op("and", at_home, op(">", var("carrying"), 0)), [emit("deposit")]),
        *return_home, assign("v_best", 1_000_000),
        {"repeat": 16, "index": "v_i", "body": [
            branch(op("and", op("<", var("v_i"), var("site_count")), op(">", site("units"), 0)), choose_site)]},
        branch(op("or", op(">=", var("carrying"), var("capacity")),
                  op("and", op(">", var("carrying"), 0),
                     op("<=", op("-", var("horizon"), var("turn")), op("+", home_distance, 4)))), return_home),
        branch(same_position, [branch(op(">=", var("v_target"), 0), [emit("harvest")], [emit("rest")])]),
        assign("v_dx", op("-", op(">", var("v_tx"), var("self_x")), op("<", var("v_tx"), var("self_x")))),
        assign("v_dy", op("-", op(">", var("v_ty"), var("self_y")), op("<", var("v_ty"), var("self_y")))),
        branch(op("and", op("!=", var("v_dx"), 0),
                  op("or", op("==", var("v_dy"), 0), op("==", op("%", var("m0"), 2), 0))),
               [emit("move", dx=var("v_dx"), dy=0)], [emit("move", dx=0, dy=var("v_dy"))])
    ]}


def reference_programs():
    return {"idle": idle_program(), "nearest": collector_program(),
            "value": collector_program(distance_weight=10, value_weight=30),
            "avoid_rival": collector_program(distance_weight=30, value_weight=10, opponent_weight=12)}
