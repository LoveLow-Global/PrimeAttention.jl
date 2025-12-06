using Primes
using Printf

"""
    calculate_efficiency(N, d_model, window, global_tokens)

Compares Standard Attention (O(N^2)) vs. Prime Attention (O(N^2 / ln(N))).
Returns the number of dot products calculated.
"""

function calculate_efficiency(N::Int; window::Int=3, global_tokens::Int=2)
    # I. Standard: Every token attends to every past token
    # Total edges = Sum(1 to N) ≈ N^2 / 2
    edges_std = (N * (N + 1)) / 2
    
    # II. Prime Attetion
    edges_prime = 0
    p_list = primes(N)
    
    for i in 1:N
        active_indices = Set{Int}()
        
        # 1. Global : 1 ... G
        for g in 1:global_tokens
            if g < i; push!(active_indices, g); end
        end
        
        # 2. Window : i-W ... i
        start_win = max(1, i - window)
        for w in start_win:i
            push!(active_indices, w)
        end
        
        # 3. Primes : i-p
        for p in p_list
            target = i - p
            if target >= 1
                push!(active_indices, target)
            end
        end
        
        edges_prime += length(active_indices)
    end
    
    return edges_std, edges_prime
end

println("Comparing Standard Causal Attention vs. Prime Attention (PrimeBird)")
@printf "%-10s | %-15s | %-15s | %-10s | %-10s\n" "Seq Len" "Standard Ops" "Prime Ops" "Speedup" "Sparsity"


for N in [2^8, 2^10, 2^12, 2^14, 2^16]
    std, prime = calculate_efficiency(N)
    
    # Calculate Metrics
    speedup = std / prime
    sparsity = 100 * (1 - (prime / std))
    
    @printf "%-10d | %-15d | %-15d | %-9.1fx | %-9.1f%%\n" N std prime speedup sparsity
end

println("Note: \"Ops\" is the number of dot-product connections computed.")
