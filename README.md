micrograd
=========

Tiny scalar-valued autograd engine with a minimal neural network library, inspired by Andrej Karpathy’s micrograd. It demonstrates how reverse‑mode automatic differentiation works end‑to‑end, with a handful of ops, a simple `Value` scalar type, and a tiny MLP built on top. too many cool words

Project Layout
--------------

- `micrograd/engine.py` — the core autograd engine (`Value` scalar, ops, backprop).
- `micrograd/nn.py` — a tiny NN library (`Neuron`, `Layers`, `MLP`).
- `test/test_engine.py` — correctness tests that compare against PyTorch.
- `setup.py` — packaging metadata.

Requirements
------------

- Python 3.8+ recommended.
- PyTorch is required only for running the tests (used as a reference implementation).

Run Tests
---------

From the repository root:

```bash
pytest -q
```

Quick Autograd Demo
-------------------

```python
from micrograd.engine import Value

x = Value(-4.0)
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()

print('y =', y.data)
print('dy/dx =', x.grad)
```

Tiny MLP Example
----------------

```python
from micrograd.engine import Value
from micrograd.nn import MLP

# 2 -> 4 -> 1 MLP
model = MLP(2, [4, 1])

# A single training sample (x: list[Value], y: float)
xs = [ [Value(2.0), Value(-1.0)], [Value(0.5), Value(1.0)] ]
ys = [ 1.0, -1.0 ]

for step in range(50):
    # forward
    ypred = [model(x) for x in xs]
    loss = sum((yp - Value(y))**2 for yp, y in zip(ypred, ys))

    # backward
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # SGD step
    for p in model.parameters():
        p.data += -0.05 * p.grad

print('loss:', loss.data)
```

API Snapshot
------------

- `Value(data, _children=(), _op='', label='')`
  - Scalar that tracks `data`, `.grad`, and a `.backward()` method.
  - Supports `+`, `-`, `*`, `/`, `**`, unary `-`, `tanh()`, `relu()`, `exp()`.
- `Module`
  - Base class with `parameters()` and `zero_grad()`.
- `Neuron(nin)` / `Layers(nin, nout)` / `MLP(nin, nouts)`
  - Callable modules returning `Value` or lists of `Value`.

Troubleshooting
---------------

- `ModuleNotFoundError: micrograd`
  - Make sure you run from the repo root and either `pip install -e .` first or set `PYTHONPATH=.` when invoking tools.
- PyTorch import/build issues
  - On Python 3.13, use nightly wheels or switch to Python 3.12.
- Tests can be run as `pytest -q`; avoid `python -m pytest` unless you set `PYTHONPATH=.`.

License and Attribution
-----------------------

- License: MIT (see LICENSE)
- Original project: Andrej Karpathy’s micrograd — https://github.com/karpathy/micrograd
- This repo contains adaptations based on the original; the MIT license and original notice are preserved.
