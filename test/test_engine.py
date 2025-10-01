import torch
from micrograd.engine import Value

def test_sanity_check():

    # ReLU variant (micrograd vs torch)
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg_relu, ymg_relu = x, y

    x = torch.Tensor([-4.0]).double(); x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt_relu, ypt_relu = x, y

    assert ymg_relu.data == ypt_relu.data.item()
    assert xmg_relu.grad == xpt_relu.grad.item()

    # tanh variant (micrograd vs torch)
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg_tanh, ymg_tanh = x, y

    x = torch.Tensor([-4.0]).double(); x.requires_grad = True
    z = 2 * x + 2 + x
    q = torch.tanh(z) + z * x
    h = torch.tanh(z * z)
    y = h + q + q * x
    y.backward()
    xpt_tanh, ypt_tanh = x, y

    assert ymg_tanh.data == ypt_tanh.data.item()
    assert xmg_tanh.grad == xpt_tanh.grad.item()

def test_more_ops():

    # ReLU variant (micrograd vs torch)
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg_relu, bmg_relu, gmg_relu = a, b, g

    a = torch.Tensor([-4.0]).double(); b = torch.Tensor([2.0]).double()
    a.requires_grad = True; b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt_relu, bpt_relu, gpt_relu = a, b, g

    tol = 1e-6
    assert abs(gmg_relu.data - gpt_relu.data.item()) < tol
    assert abs(amg_relu.grad - apt_relu.grad.item()) < tol
    assert abs(bmg_relu.grad - bpt_relu.grad.item()) < tol

    # tanh variant (micrograd vs torch)
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg_tanh, bmg_tanh, gmg_tanh = a, b, g

    a = torch.Tensor([-4.0]).double(); b = torch.Tensor([2.0]).double()
    a.requires_grad = True; b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + torch.tanh(b + a)
    d = d + 3 * d + torch.tanh(b - a)
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt_tanh, bpt_tanh, gpt_tanh = a, b, g

    assert abs(gmg_tanh.data - gpt_tanh.data.item()) < tol
    assert abs(amg_tanh.grad - apt_tanh.grad.item()) < tol
    assert abs(bmg_tanh.grad - bpt_tanh.grad.item()) < tol


def test_extreme_values():
    tol = 1e-9

    # tanh saturates for large magnitudes; gradients ~ 0
    for v in [20.0, -20.0, 10.0, -10.0, 0.0]:
        # micrograd
        xm = Value(v)
        ym = xm.tanh()
        ym.backward()

        # torch
        xt = torch.tensor([v], dtype=torch.float64, requires_grad=True)
        yt = torch.tanh(xt)
        yt.backward(torch.ones_like(yt))

        assert abs(ym.data - yt.item()) < tol
        assert abs(xm.grad - xt.grad.item()) < tol

    # ReLU: large positive passes through, large negative zeros out, zero edge
    for v in [1e9, -1e9, 0.0, 3.14, -2.71]:
        # micrograd
        xm = Value(v)
        ym = xm.relu()
        ym.backward()

        # torch
        xt = torch.tensor([v], dtype=torch.float64, requires_grad=True)
        yt = torch.relu(xt)
        yt.backward(torch.ones_like(yt))

        assert abs(ym.data - yt.item()) < tol
        assert abs(xm.grad - xt.grad.item()) < tol
