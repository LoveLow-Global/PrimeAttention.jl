"""
    prime_attention_kernel(q, k, v, primes_list, window, global_tokens)

Compute Causal PrimeBird attention:
1. Global: All tokens attend to the first `global_tokens`.
2. Window: All tokens attend to the last `window` neighbors.
3. Prime: Tokens attend to `i - p` (skipping if covered by Global/Window).
"""
function prime_attention_kernel(q::AbstractMatrix{T}, 
                                k::AbstractMatrix{T}, 
                                v::AbstractMatrix{T}, 
                                primes_list::Vector{Int},
                                window::Int,
                                global_tokens::Int) where T
    
    d_head, seq_len = size(q)
    scale = T(1.0 / sqrt(d_head))
    
    # Trackable buffer for Zygote
    y_buf = Zygote.Buffer(q) 
    
    @inbounds for i in 1:seq_len
        # Max Score
        max_score = T(-Inf)
        
        # 1. Global Attention
        # Only if i > global_tokens (otherwise covered by self/window loop)
        for g in 1:global_tokens
            if g >= i; break; end # Causal: does not look forward
            
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, g]; end
            max_score = max(max_score, score * scale)
        end

        # 2. Local Window
        # This includes Self-Attention
        start_win = max(1, i - window)
        for j in start_win:i
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, j]; end
            max_score = max(max_score, score * scale)
        end

        # 3. Prime Distances
        for p in primes_list
            j = i - p
            if j < 1; break; end
            
            # Overlap Check
            if j <= global_tokens || j >= (i - window)
                continue 
            end
            
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, j]; end
            max_score = max(max_score, score * scale)
        end
        
        # Weighted Sum
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

        # 2. Window
        start_win = max(1, i - window)
        for j in start_win:i
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, j]; end
            weight = exp((score * scale) - max_score)
            sum_exp += weight
            
            for d in 1:d_head; y_buf[d, i] += v[d, j] * weight; end
        end

        # 3. Prime
        for p in primes_list
            j = i - p
            if j < 1; break; end
            
            if j <= global_tokens || j >= (i - window)
                continue 
            end
            
            score = zero(T)
            for d in 1:d_head; score += q[d, i] * k[d, j]; end
            weight = exp((score * scale) - max_score)
            sum_exp += weight
            
            for d in 1:d_head; y_buf[d, i] += v[d, j] * weight; end
        end
        
        inv_sum = T(1.0) / sum_exp
        for d in 1:d_head
            y_buf[d, i] *= inv_sum
        end
    end
    
    return copy(y_buf)
end