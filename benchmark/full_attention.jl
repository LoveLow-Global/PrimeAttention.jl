using Lux

"""
    StandardFullAttention(dims::Int; heads = 1)

Baseline causal multi-head self-attention over all previous tokens: standard O(N^2)
attention, computed with batched matrix multiplication (BLAS) for a fair comparison.
It has the same parameter layout as `SparseIndexAttention`, so both can share parameters.
"""
struct StandardFullAttention{W1, W2} <: Lux.AbstractLuxContainerLayer{(:Wi, :Wo)}
    heads::Int
    Wi::W1
    Wo::W2
end

function StandardFullAttention(dims::Int; heads = 1)
    return StandardFullAttention(heads, Dense(dims => dims * 3), Dense(dims => dims))
end

function (m::StandardFullAttention)(x::AbstractArray{T, 3}, ps, st) where {T}
    _, seq_len, batch_size = size(x)

    qkv, st_Wi = m.Wi(x, ps.Wi, st.Wi)
    chunk = div(size(qkv, 1), 3)
    d_head = div(chunk, m.heads)

    # (chunk, seq_len, batch) -> (d_head, seq_len, heads * batch), one slice per head
    function split_heads(a)
        a = reshape(a, d_head, m.heads, seq_len, batch_size)
        return reshape(permutedims(a, (1, 3, 2, 4)), d_head, seq_len, :)
    end
    q = split_heads(qkv[1:chunk, :, :])
    k = split_heads(qkv[(chunk + 1):(2 * chunk), :, :])
    v = split_heads(qkv[(2 * chunk + 1):end, :, :])

    # scores[j, i] = k_j ⋅ q_i / sqrt(d_head), with keys after the query masked out
    scores = batched_mul(batched_transpose(k), q) ./ sqrt(T(d_head))
    causal = [j <= i for j in 1:seq_len, i in 1:seq_len]
    weights = softmax(ifelse.(causal, scores, T(-Inf)); dims = 1)

    # Merge the heads back into (chunk, seq_len, batch)
    y = reshape(batched_mul(v, weights), d_head, seq_len, m.heads, batch_size)
    out_concat = reshape(permutedims(y, (1, 3, 2, 4)), chunk, seq_len, batch_size)

    out, st_Wo = m.Wo(out_concat, ps.Wo, st.Wo)
    return out, (Wi = st_Wi, Wo = st_Wo)
end
