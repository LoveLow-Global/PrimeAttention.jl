"""
    SparseIndexAttention(heads, dims, indices_list, window, global_tokens, Wi, Wo)

A generic causal sparse multi-head attention layer. The embedding of size `dims` is split
into `heads` heads of size `dims ÷ heads`, and in each head token `i` attends to:
1. Global tokens (hubs): the first `global_tokens` tokens.
2. Local window (neighbors): tokens `i - window` to `i`.
3. Sparse indices: tokens `i - s` for each `s` in `indices_list`.

`Wi` projects the input to queries, keys and values (`dims => 3 * dims`), and `Wo` projects
the concatenated heads back (`dims => dims`). Usually built with [`PrimeSelfAttention`](@ref),
[`SquareSelfAttention`](@ref) or [`MianChowlaSelfAttention`](@ref).
"""
struct SparseIndexAttention{W1, W2} <: Lux.AbstractLuxContainerLayer{(:Wi, :Wo)}
    heads::Int
    dims::Int
    indices_list::Vector{Int}
    window::Int
    global_tokens::Int
    Wi::W1
    Wo::W2

    function SparseIndexAttention(
            heads,
            dims,
            indices_list,
            window,
            global_tokens,
            Wi::W1,
            Wo::W2,
        ) where {W1, W2}
        if heads < 1 || dims % heads != 0
            throw(ArgumentError("`dims` ($dims) must be divisible by `heads` ($heads)"))
        end
        return new{W1, W2}(heads, dims, indices_list, window, global_tokens, Wi, Wo)
    end
end

# Generic Forward Pass
function (m::SparseIndexAttention)(x::AbstractArray{T, 3}, ps, st) where {T}
    dims_in, seq_len, batch_size = size(x)

    # Pass x through Wi using its specific parameters and state
    qkv, st_Wi = m.Wi(x, ps.Wi, st.Wi)
    dims_qkv = size(qkv, 1)
    chunk = div(dims_qkv, 3)
    d_head = div(chunk, m.heads)

    # Pre-allocate for type stability
    out_concat = similar(qkv, chunk, seq_len, batch_size)

    # Each head attends independently within its own rows of q, k and v
    for b in 1:batch_size, h in 1:m.heads
        rows = ((h - 1) * d_head + 1):(h * d_head)
        q_slice = view(qkv, rows, :, b)
        k_slice = view(qkv, rows .+ chunk, :, b)
        v_slice = view(qkv, rows .+ 2 * chunk, :, b)
        y_slice = view(out_concat, rows, :, b) # View into the pre-allocated output

        sparse_index_kernel!(
            y_slice,
            q_slice,
            k_slice,
            v_slice,
            m.indices_list,
            m.window,
            m.global_tokens,
        )
    end

    # Pass through Wo and capture updated state
    out, st_Wo = m.Wo(out_concat, ps.Wo, st.Wo)

    new_st = (Wi = st_Wi, Wo = st_Wo)

    return out, new_st
end

# Constructors

"""
    PrimeSelfAttention(dims::Int; heads = 1, window = 3, global_tokens = 2, seq_len_max = 2^12)

Sparse self-attention whose long-range connections skip back by prime distances
(2, 3, 5, 7, ...). Takes input of size `(dims, seq_len, batch)` and returns output of the
same size. See [`SparseIndexAttention`](@ref) for the full attention pattern.

# Keywords
- `heads`: number of attention heads; must divide `dims`.
- `window`: each token attends to the `window` tokens before it.
- `global_tokens`: number of leading tokens that every token attends to.
- `seq_len_max`: largest distance covered by the sparse connections.
"""
function PrimeSelfAttention(
        dims::Int;
        heads = 1,
        window = 3,
        global_tokens = 2,
        seq_len_max = 2^12,
    )
    p_list = primes(seq_len_max)
    wi = Dense(dims => dims * 3)
    wo = Dense(dims => dims)
    return SparseIndexAttention(heads, dims, p_list, window, global_tokens, wi, wo)
end

"""
    SquareSelfAttention(dims::Int; heads = 1, window = 3, global_tokens = 2, seq_len_max = 2^12)

Sparse self-attention whose long-range connections skip back by square distances
(1, 4, 9, 16, ...), giving `O(N√N)` cost. Keywords are the same as for
[`PrimeSelfAttention`](@ref).
"""
function SquareSelfAttention(
        dims::Int;
        heads = 1,
        window = 3,
        global_tokens = 2,
        seq_len_max = 2^12,
    )
    s_list = [i^2 for i in 1:isqrt(seq_len_max)]
    wi = Dense(dims => dims * 3)
    wo = Dense(dims => dims)
    return SparseIndexAttention(heads, dims, s_list, window, global_tokens, wi, wo)
end

"""
    MianChowlaSelfAttention(dims::Int; heads = 1, window = 3, global_tokens = 2, seq_len_max = 2^12)

Sparse self-attention whose long-range connections skip back by Mian-Chowla distances
(1, 2, 4, 8, 13, ...), a greedy Sidon set in which all pairwise sums are distinct.
Keywords are the same as for [`PrimeSelfAttention`](@ref).
"""
function MianChowlaSelfAttention(
        dims::Int;
        heads = 1,
        window = 3,
        global_tokens = 2,
        seq_len_max = 2^12,
    )
    # Mian-Chowla sequence (greedy Sidon set) generation
    m_list = Int[1]
    sums = Set{Int}([2])
    for n in 2:seq_len_max
        new_sums = [n + x for x in m_list]
        push!(new_sums, 2n)
        if all(s -> !(s in sums), new_sums)
            push!(m_list, n)
            union!(sums, new_sums)
        end
    end
    wi = Dense(dims => dims * 3)
    wo = Dense(dims => dims)
    return SparseIndexAttention(heads, dims, m_list, window, global_tokens, wi, wo)
end
