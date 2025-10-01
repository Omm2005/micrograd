import random
import math

from micrograd.engine import Value
from micrograd.nn import MLP


def test_mlp_parameter_count_and_shapes():
    # MLP(2, [3, 1]) should have: (2*3 + 3) + (3*1 + 1) = 9 + 4 = 13 params
    random.seed(42)
    model = MLP(2, [3, 1])
    params = model.parameters()
    assert len(params) == (2 * 3 + 3) + (3 * 1 + 1)

    # Forward shape: last layer has 1 neuron -> returns a single Value
    out = model([Value(1.0), Value(-2.0)])
    assert isinstance(out, Value)


def test_mlp_learns_simple_regression():
    # Deterministic init
    random.seed(123)

    # Simple dataset: y = 0.5*x1 - x2 (noiseless)
    data = [
        ([Value(2.0), Value(-1.0)], 0.5*2.0 - (-1.0)),
        ([Value(-1.0), Value(2.0)], 0.5*(-1.0) - 2.0),
        ([Value(0.5), Value(1.0)], 0.5*0.5 - 1.0),
        ([Value(1.5), Value(-0.5)], 0.5*1.5 - (-0.5)),
    ]

    model = MLP(2, [4, 1])

    # compute initial loss
    def loss_fn():
        preds = [model(x) for x, _ in data]
        target_vals = [Value(y) for _, y in data]
        return sum((p - t) ** 2 for p, t in zip(preds, target_vals))

    loss0 = loss_fn().data

    # train a few steps
    for _ in range(100):
        for p in model.parameters():
            p.grad = 0.0
        loss = loss_fn()
        loss.backward()
        for p in model.parameters():
            p.data += -0.05 * p.grad

    loss1 = loss_fn().data

    # Expect improvement
    assert loss1 < loss0

