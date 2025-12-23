"""
    SparseIndexAttention(dims::Int, indices::Vector{Int}, heads::Int, window::Int, global_tokens::Int)

A generic sparse attention layer that attends to:
1. Global tokens (hubs).
2. Local window (neighbors).
3. Sparse indices provided in `indices_list`.
"""
struct SparseIndexAttention{D}
    heads::Int
    dims::Int
    indices_list::Vector{Int}
    window::Int
    global_tokens::Int
    Wi::D
    Wo::D
end

Flux.@layer SparseIndexAttention

# Generic Forward Pass
function (m::SparseIndexAttention)(x::AbstractArray{T,3}) where {T}
    dims_in, seq_len, batch_size = size(x)
    qkv = m.Wi(x)
    chunk = div(size(qkv, 1), 3)

    outputs = map(1:batch_size) do b
        q_slice = view(qkv, 1:chunk, :, b)
        k_slice = view(qkv, (chunk+1):(2*chunk), :, b)
        v_slice = view(qkv, (2*chunk+1):(3*chunk), :, b)

        # Use the universal kernel
        sparse_index_kernel(
            q_slice,
            k_slice,
            v_slice,
            m.indices_list,
            m.window,
            m.global_tokens,
        )
    end

    out_concat = cat(outputs..., dims = 3)
    return m.Wo(out_concat)
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
    wi = Dense(dims => dims*3)
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
    s_list = [i^2 for i = 1:isqrt(seq_len_max)]
    wi = Dense(dims => dims*3)
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
    for n = 2:seq_len_max
        new_sums = [n + x for x in m_list]
        push!(new_sums, 2n)
        if all(s -> !(s in sums), new_sums)
            push!(m_list, n)
            union!(sums, new_sums)
        end
    end
    wi = Dense(dims => dims*3)
    wo = Dense(dims => dims)
    return SparseIndexAttention(heads, dims, m_list, window, global_tokens, wi, wo)
end
