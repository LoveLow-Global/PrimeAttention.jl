"""
    PrimeSelfAttention(dims::Int; heads=1, seq_len_max=2^12, window=3, global_tokens=2)

Causal "PrimeBird" Attention Layer.
- Global : Attends to first `global_tokens` (Hubs).
- Window : Attends to `window` neighbors (Local).
- Prime : Attends to prime intervals (Sparse Long-Range).
"""
struct PrimeSelfAttention{D}
    heads::Int
    dims::Int
    primes_list::Vector{Int}
    window::Int
    global_tokens::Int
    Wi::D
    Wo::D
end

Flux.@layer PrimeSelfAttention

function PrimeSelfAttention(dims::Int, heads::Int, window::Int, global_tokens::Int; seq_len_max::Int=2^12)
    
    p_list = primes(seq_len_max)
    wi = Dense(dims => dims*3)
    wo = Dense(dims => dims)
    
    return PrimeSelfAttention(
        heads,
        dims,
        p_list,
        window,
        global_tokens,
        wi,
        wo
    )
end

function (m::PrimeSelfAttention)(x::AbstractArray{T, 3}) where T
    dims_in, seq_len, batch_size = size(x)
    
    qkv = m.Wi(x) 
    chunk = div(size(qkv, 1), 3)
    
    outputs = map(1:batch_size) do b
        q_slice = view(qkv, 1:chunk, :, b)
        k_slice = view(qkv, chunk+1:2*chunk, :, b)
        v_slice = view(qkv, 2*chunk+1:3*chunk, :, b)
        
        prime_attention_kernel(
            q_slice, k_slice, v_slice, 
            m.primes_list, 
            m.window, 
            m.global_tokens
        )
    end
    
    out_concat = cat(outputs..., dims=3)
    return m.Wo(out_concat)
end