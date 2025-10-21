import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pprint import pprint as pp
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 3.3 Self-attention without trainable weights
    - Attention scores computed as dot product of query tensor and input tensors
    - Dot product is a measure of similarity--it measures how closely two tensors are aligned
      - So attention in this simple example is just based on token embedding similarity
    - Attention scores are normalized into attention weights via softmax
      - Softmax takes list of real numbers and converts them to probabilities that sum to one
      - Larger values are exponentially amplified and smaller values are suppressed
      - Called softmax because it's a "softened" version of hard max, which sets the largest value to 1 and the rest to 0
        - Hard max is not differentiable
    - Context tensor is computed by multiplying each input tensor by its attention weight and summing
      - So the context vector is an attention-weighted sum of all input tensors
      - Input tensors with higher tension will make larger contributions to the context vector value
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
    mo.md(r"""## Compute context vector for single input token""")
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
    mo.md(r"""## Compute context vector for all input tokens""")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 3.4 Self-attention with trainable weights
    - Basic self-attention with trainable weights is called *scaled dot-product attention*
    - Introduce three trainable weight matrices $W_q$, $W_k$, and $W_v$ which represent query, key, and value matrices respectively
    - These matrices serve to project input embeddings into query, key, and value spaces respectively
    - Steps to compute context vector for query input $x^{(n)}$:
      1. Compute query vector for query input only as $q^{(n)} = W_q \cdot x^{(n)}$ 
      2. Compute key and value vectors for all inputs as:
        - $k^{(i)} = W_k \cdot x^{(i)}$
        - $v^{(i)} = W_v \cdot x^{(i)}$
      3. Compute attention scores as dot product between input query vector and each key vector
        - $\omega_{ni} = q^{(n)} \cdot k^{(i)}$
    > Contrast this against the previous mechanism which simple computed the attention score as the dot product of the current input vector with each other input vector
      4. Compute attention weights by scaling attention scores by the square root of the embedding dimension and then applying *softmax*
        - $\alpha_{ni} = softmax(\omega_{ni} / \sqrt{d_k})$
    > Why do we normalize by the embedding dimension size?
    >
    > The high level reason is to improve training peformance by avoiding small gradients. But why is this at the mechanical level?
      5. Compute context vector as sum of value vectors weighted by attention score
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Compute context vector for single input token""")
    return


@app.cell
def _(inputs):
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    print(f"{x_2=}")
    print(f"{d_in=}")
    print(f"{d_out=}")
    return d_in, d_out, x_2


@app.cell
def _(d_in, d_out, torch):
    torch.manual_seed(123)  # For reproducible results

    W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    print(f"{W_q=}")
    print(f"{W_k=}")
    print(f"{W_v=}")
    return W_k, W_q, W_v


@app.cell
def _(W_k, W_q, W_v, x_2):
    q_2 = x_2 @ W_q
    k_2 = x_2 @ W_k
    v_2 = x_2 @ W_v

    print(f"{q_2=}")
    print(f"{k_2=}")
    print(f"{v_2=}")
    return (q_2,)


@app.cell
def _(inputs):
    inputs
    return


@app.cell
def _(W_k, W_v, inputs):
    keys = inputs @ W_k
    values = inputs @ W_v

    print(f"{keys=}")
    print(f"{values=}")
    print(f"{keys.shape=}")
    print(f"{values.shape=}")
    return keys, values


@app.cell
def _(keys, q_2):
    sdp_attn_scores_2 = q_2 @ keys.T
    print(f"{sdp_attn_scores_2=}")
    return (sdp_attn_scores_2,)


@app.cell
def _(keys, sdp_attn_scores_2, torch):
    embed_dim = keys.shape[-1]
    scaled_attn_weights_2 = sdp_attn_scores_2 / embed_dim**0.5
    sdp_attn_weights_2 = torch.softmax(scaled_attn_weights_2, dim=-1)
    print(f"{sdp_attn_weights_2=}")
    return (sdp_attn_weights_2,)


@app.cell
def _(sdp_attn_weights_2, values):
    context_vec_2 = sdp_attn_weights_2 @ values
    print(f"{context_vec_2=}")
    return (context_vec_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Compute context vector for all input tokens""")
    return


@app.cell
def _(torch):
    import torch.nn as nn


    class SelfAttention_v1(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        def forward(self, x):
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value
            attn_scores = queries @ keys.T  # omega
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1] ** 0.5, dim=-1
            )
            context_vec = attn_weights @ values
            return context_vec
    return SelfAttention_v1, nn


@app.cell
def _(SelfAttention_v1, context_vec_2, d_in, d_out, inputs, torch):
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    context_vec = sa_v1(inputs)
    assert torch.allclose(context_vec[1], context_vec_2)
    print(f"{context_vec=}")
    return


@app.cell
def _(nn, torch):
    # Improvement upon v1 self-attention with nn.Linear, which implements y = M @ X + b, which is effectively a normal matrix mult when bias term is disabled. nn.Linear also automatically initializes weights with an optimized weight init scheme, leading to more stable and effective model training.
    class SelfAttention_v2(nn.Module):
        def __init__(self, d_in, d_out, qkv_bias=False):
            super().__init__()
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        def forward(self, x):
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)
            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1] ** 0.5, dim=-1
            )
            context_vec = attn_weights @ values
            return context_vec
    return (SelfAttention_v2,)


@app.cell
def _(SelfAttention_v2, d_in, d_out, inputs, torch):
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))
    return (sa_v2,)


@app.cell
def _(sa_v2):
    sa_v2.W_query._parameters["weight"]
    return


@app.cell
def _(SelfAttention_v1, SelfAttention_v2, d_in, d_out, inputs, nn, torch):
    def validate():
        """
        Validate that SA v1 and v2 produce the same results when inited with the same weights
        """
        sa_v1 = SelfAttention_v1(d_in, d_out)
        sa_v2 = SelfAttention_v2(d_in, d_out)

        # Inject v2 weights to v2
        sa_v1.W_query = nn.Parameter(sa_v2.W_query._parameters["weight"].T)
        sa_v1.W_key = nn.Parameter(sa_v2.W_key._parameters["weight"].T)
        sa_v1.W_value = nn.Parameter(sa_v2.W_value._parameters["weight"].T)

        c_v1 = sa_v1(inputs)
        c_v2 = sa_v2(inputs)

        torch.allclose(c_v1, c_v2)


    validate()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 3.5 Hiding future words with causal attention
    - Only want to consider tokens that appear prior to the current position when predicting the next token
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3.5.1 Applying a causal attention mask
    - Naive masking method:
      1. Normalize attention scores into attention weights with softmax
      2. Mask attention weights above diagonal with 0's
      3. Re-normalize non-masked elements so each row sums to 1
    - Efficient masking method
      1. Mask attention scores above diagonal with -inf's
      2. Normalized masked attention scores into attention weights with softmax
    - The efficient method works because softmax converts inputs into a probability distribution and -inf values are treated as zero probability
    """
    )
    return


@app.cell
def _(inputs, sa_v2, torch):
    def compute_masked_attention_weights_naive():
        queries = sa_v2.W_query(inputs)
        keys = sa_v2.W_key(inputs) 
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        print(f"{attn_weights=}")

        context_length = attn_scores.shape[0]
        mask_simple = torch.tril(torch.ones(context_length, context_length))
        print(f"{mask_simple=}")

        masked_simple = attn_weights*mask_simple
        print(f"{masked_simple=}")

        row_sums = masked_simple.sum(dim=-1, keepdim=True)
        masked_simple_norm = masked_simple / row_sums
        print(f"{masked_simple_norm}")

    compute_masked_attention_weights_naive()
    return


@app.cell
def _(inputs, sa_v2, torch):
    def compute_masked_attention_weights_efficient():
        queries = sa_v2.W_query(inputs)
        keys = sa_v2.W_key(inputs) 
        attn_scores = queries @ keys.T
    
        context_length = attn_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        print(f"{mask=}")
    
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        print(f"{masked=}")
    
        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
        print(f"{attn_weights}")

    compute_masked_attention_weights_efficient()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3.5.2 Masking additional attention weights with dropout
    - Dropout is a technique in the training stage of deep learning where hidden layer elements are randomly ignored--that is "dropped out"
    - This technique helps to avoid overfitting by preventing the model from becoming too reliant on specific hidden layer units
    - In attention heads, dropout is usually applied in one of two places:
      1. After attention weights are computed
      2. After applying attention weights to value vectors
    - Option 1 is the more common variant so this is what we will use
    - After an X% dropout is applied to attention weights, the remaining weights must be scalled up by a factor of 100/X
      - This is essential to preserve the total attention mass
    """
    )
    return


@app.cell
def _(attn_weights, torch):
    def apply_dropout():
        torch.manual_seed(123)
        dropout = torch.nn.Dropout(0.5)
        dropout_attn_weights = dropout(attn_weights)
        print(f"{dropout_attn_weights=}")
    apply_dropout()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3.5.3 Implementing a compact causal attention class
    - Let's put it all together and implement a single, compact attention class
    """
    )
    return


@app.cell
def _(inputs, torch):
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)
    return (batch,)


@app.cell
def _(nn, torch):
    class CausalAttention(nn.Module):
        def __init__(self, d_in, d_out, context_length,
                    dropout, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.dropout = nn.Dropout(dropout)
            # NOTE: register buffer should be used for constant tensors.
            # Among other things, it ensures that the constants are moved between devices appropriately.
            self.register_buffer(
               'mask',
               torch.triu(torch.ones(context_length, context_length),
               diagonal=1)
            )

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.transpose(1, 2)   
            attn_scores.masked_fill_(
                self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1
            )
            attn_weights = self.dropout(attn_weights)

            context_vec = attn_weights @ values
            return context_vec
    return (CausalAttention,)


@app.cell
def _(CausalAttention, batch, d_in, d_out, torch):
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
