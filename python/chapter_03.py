import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Section 3.3
    - Attention scores computed as dot product of query tensor and input tensors
    - Dot product is a measure of similarity--it measures how closely two tensors are aligned
    - Attention scores are normalized into attention weights via softmax
    - Context tensor is computed by multiplying each input tensor by its attention weight and summing
      - What is the context vector in intuitive terms?
    """
    )
    return


@app.cell
def _():
    import torch

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]
    )
    return inputs, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Compute attention weights for single input vector""")
    return


@app.cell
def _(inputs, torch):
    # Compute attention scores as dot product of query tensor and input tensors
    def compute_attn_scores_single(query_idx, inputs):
        query = inputs[query_idx]
        attn_scores = torch.empty(inputs.shape[0])
        for i, x_i in enumerate(inputs):
            attn_scores[i] = torch.dot(x_i, query)
        return attn_scores


    attn_scores_2 = compute_attn_scores_single(1, inputs)
    print(f"Attention scores: {attn_scores_2}")
    return (attn_scores_2,)


@app.cell
def _(attn_scores_2, torch):
    # Normalize attention scores into attention weights via softmax
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print(f"Attention weights: {attn_weights_2}")
    print(f"Sum: {attn_weights_2.sum()}")
    return (attn_weights_2,)


@app.cell
def _(attn_weights_2, inputs, torch):
    # Compute context tensor as summation of all input tensors multiplied by their coorresponding attention weight
    def compute_context_single(inputs, attn_weights):
        context = torch.zeros(inputs.shape[1])
        for i, x_i in enumerate(inputs):
            context += x_i * attn_weights[i]
        return context


    context_2 = compute_context_single(inputs, attn_weights_2)
    print(f"Context tensor: {context_2}")
    return (context_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Compute attention weights for all input tokens""")
    return


@app.cell
def _(inputs):
    def compute_attn_scores(inputs):
        return inputs @ inputs.T


    attn_scores = compute_attn_scores(inputs)
    print(attn_scores)
    return (attn_scores,)


@app.cell
def _(attn_scores, torch):
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(f"Attenion weights: {attn_weights}")
    print(f"Attenion sum: {attn_weights.sum(dim=-1)}")
    return (attn_weights,)


@app.cell
def _(attn_weights, context_2, inputs, torch):
    def compute_context(inputs, attn_weights):
        return attn_weights @ inputs


    context = compute_context(inputs, attn_weights)
    assert torch.allclose(context_2, context[2 - 1])
    print(context)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
