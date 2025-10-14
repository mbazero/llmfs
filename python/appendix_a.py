import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## A.4 Automatic differentiation made easy""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    from torch.autograd import grad

    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1 * w1 + b
    a = torch.sigmoid(z)

    loss = F.binary_cross_entropy(a, y)

    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b = grad(loss, b, retain_graph=True)
    return a, b, grad, grad_L_b, grad_L_w1, w1, z


@app.cell
def _(grad_L_b, grad_L_w1):
    print(grad_L_w1)
    print(grad_L_b)
    return


@app.cell
def _(a, grad, grad_L_w1, w1, z):
    print(grad(w1, w1, retain_graph=True))
    print(grad(z, w1, retain_graph=True))
    print(grad(a, w1, retain_graph=True))
    print(grad_L_w1)
    return


@app.cell
def _(a, b, grad, grad_L_b, z):
    print(grad(b, b, retain_graph=True))
    print(grad(z, b, retain_graph=True))
    print(grad(a, b, retain_graph=True))
    print(grad_L_b)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
