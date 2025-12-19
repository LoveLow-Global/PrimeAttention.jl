"""
    sparse_index_kernel(q, k, v, indices_list, window, global_tokens)

General kernel for Sparse Attention.
"""
function sparse_index_kernel(q::AbstractMatrix{T}, 
                             k::AbstractMatrix{T}, 
                             v::AbstractMatrix{T}, 
                             indices_list::Vector{Int},
                             window::Int,
                             global_tokens::Int) where T
    
    d_head, seq_len = size(q)
    scale = T(1.0 / sqrt(d_head))
    y_buf = Zygote.Buffer(q) 
    
    @inbounds for i in 1:seq_len
        max_score = T(-Inf)
        
        # 1. Global Attention
        for g in 1:global_tokens
            if g >= i; break; end 
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, g]; end
            max_score = max(max_score, score * scale)
        end

        # 2. Local Window
        start_win = max(1, i - window)
        for j in start_win:i
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, j]; end
            max_score = max(max_score, score * scale)
        end

        # 3. Sparse Indices (Primes, Squares, or Mian-Chowla)
        for idx in indices_list
            j = i - idx
            if j < 1; break; end
            
            # Skip if already handled
            if j <= global_tokens || j >= (i - window)
                continue 
            end
            
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, j]; end
            max_score = max(max_score, score * scale)
        end
        
        sum_exp = zero(T)
        for d in 1:d_head; y_buf[d, i] = zero(T); end

        # 1. Global
        for g in 1:global_tokens
            if g >= i; break; end
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, g]; end
            weight = exp((score * scale) - max_score)
            sum_exp += weight
            for d in 1:d_head; y_buf[d, i] += v[d, g] * weight; end
        end

        # 2. Window (Neighbour)
        start_win = max(1, i - window)
        for j in start_win:i
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, j]; end
            weight = exp((score * scale) - max_score)
            sum_exp += weight
            for d in 1:d_head; y_buf[d, i] += v[d, j] * weight; end
        end

        # 3. Sparse Indices
        for idx in indices_list
            j = i - idx
            if j < 1; break; end
            if j <= global_tokens || j >= (i - window); continue; end
            
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, j]; end
            weight = exp((score * scale) - max_score)
            sum_exp += weight
            for d in 1:d_head; y_buf[d, i] += v[d, j] * weight; end
        end
        
        inv_sum = T(1.0) / sum_exp
        for d in 1:d_head; y_buf[d, i] *= inv_sum; end
    end
    
    return copy(y_buf)
end