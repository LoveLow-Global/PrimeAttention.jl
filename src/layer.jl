"""
SparseIndexAttention(dims::Int, indices::Vector{Int}, heads::Int, window::Int, global_tokens::Int)

A generic sparse attention layer that attends to:
1. Global tokens (hubs).
2. Local window (neighbors).
3. Sparse indices provided in `indices_list`.
"""
struct SparseIndexAttention{W1, W2} <: Lux.AbstractLuxContainerLayer{(:Wi, :Wo)}
    heads::Int
    dims::Int
    indices_list::Vector{Int}
    window::Int
    global_tokens::Int
    Wi::W1
    Wo::W2
end

# Generic Forward Pass
function (m::SparseIndexAttention)(x::AbstractArray{T, 3}, ps, st) where {T}
    dims_in, seq_len, batch_size = size(x)

    # Pass x through Wi using its specific parameters and state
    qkv, st_Wi = m.Wi(x, ps.Wi, st.Wi)
    dims_qkv = size(qkv, 1)
    chunk = div(dims_qkv, 3)

    # Pre-allocate for type stability
    out_concat = similar(qkv, chunk, seq_len, batch_size)

    for b in 1:batch_size
        q_slice = view(qkv, 1:chunk, :, b)
        k_slice = view(qkv, (chunk + 1):(2 * chunk), :, b)
        v_slice = view(qkv, (2 * chunk + 1):(3 * chunk), :, b)
        y_slice = view(out_concat, :, :, b) # View into the pre-allocated output

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
