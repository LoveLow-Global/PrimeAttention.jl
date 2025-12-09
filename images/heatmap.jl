# Heatmap of Attentions for PrimeAttention.

# The example below is when N = 200.
using Plots
using Primes
using LinearAlgebra
using Statistics

# Pattern Generator
function get_prime_attention_scores(seq_len::Int)

    masked_scores = zeros(Float32, seq_len, seq_len)
    
    # Generate random scores to simulate attention intensity
    # In a real model, these would be the softmax probabilities
    active_scores = rand(Float32, seq_len, seq_len)
    
    # Get Prime indices
    p_list = primes(seq_len)
    
    for i in 1:seq_len
        # 1. Self-Attention (Diagonal) - Always active
        masked_scores[i, i] = active_scores[i, i]
        
        # 2. Prime Lookback
        for p in p_list
            j = i - p
            if j < 1; break; end
            
            # Set the score for the valid Prime connection
            # i = Query (Target), j = Key (Source)
            masked_scores[i, j] = active_scores[i, j]
        end
    end
    
    return masked_scores
end

# Configuration
seq_len = 200
matrix = get_prime_attention_scores(seq_len)

# Plotting
default(size=(700, 600), margin=5Plots.mm)

heatmap(
    matrix,
    c=:viridis,
    yflip=true,    # (1,1) at top-left, like a matrix
    clims=(0, 1),  # 0 to be the darkest color
    title="PrimeAttention Scores (Head 1)",
    xlabel="Key Index (Source)",
    ylabel="Query Index (Target)",
    framestyle=:box,
    grid=false
)

savefig("prime_attention_heatmap.png")