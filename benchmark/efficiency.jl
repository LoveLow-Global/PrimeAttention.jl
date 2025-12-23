using PrimeAttention
using Flux
using BenchmarkTools
using Printf

"""
    StandardFullAttention(dims::Int; heads=1)

A baseline implementation of standard O(N^2) Full Attention using 
efficient batched matrix multiplication (BLAS) for fair comparison.
"""
struct StandardFullAttention
    dims::Int
    scale::Float32
    Wi::Dense
    Wo::Dense
end

Flux.@layer StandardFullAttention

function StandardFullAttention(dims::Int)
    Wi = Dense(dims => dims*3)
    Wo = Dense(dims => dims)
    scale = 1.0f0 / sqrt(Float32(dims))
    return StandardFullAttention(dims, scale, Wi, Wo)
end

function (m::StandardFullAttention)(x::AbstractArray{T,3}) where {T}
    dims, seq_len, batch_size = size(x)

    qkv = m.Wi(x)
    chunk = div(dims*3, 3)

    q = view(qkv,(1:chunk),:,:)
    k = view(qkv,((chunk+1):(2*chunk)),:,:)
    v = view(qkv,((2*chunk+1):(3*chunk)),:,:)

    scores = NNlib.batched_mul(permutedims(q, (2, 1, 3)), k) .* m.scale

    weights = softmax(scores, dims = 2)

    context = NNlib.batched_mul(v, permutedims(weights, (2, 1, 3)))

    return m.Wo(context)
end

# Configuration
const DIMS = 64      # Embedding dimension
const SEQ_LEN = 2048 # Sequence length (N)
const BATCH_SIZE = 4 # Batch size
const GLOBAL = 3     # Global tokens
const WINDOW = 3     # Local window size
const HEADS = 1      # Heads (Single head for this test)

@printf("Config: Sequence Length=%d, Batch=%d, Dims=%d\n", SEQ_LEN, BATCH_SIZE, DIMS)

# Dummy Input
x = randn(Float32, DIMS, SEQ_LEN, BATCH_SIZE)

# Initialize Models
full_attn = StandardFullAttention(DIMS)
prime_attn =
    PrimeSelfAttention(DIMS, window = WINDOW, global_tokens = GLOBAL, seq_len_max = SEQ_LEN)
sq_attn = SquareSelfAttention(
    DIMS,
    window = WINDOW,
    global_tokens = GLOBAL,
    seq_len_max = SEQ_LEN,
)
mc_attn = MianChowlaSelfAttention(
    DIMS,
    window = WINDOW,
    global_tokens = GLOBAL,
    seq_len_max = SEQ_LEN,
)

function run_benchmark(name, model, input)
    println("Benchmarking $name")
    model(input)
    b = @benchmark $model($input)

    t_med = median(b.times) / 1e6

    @printf("Median Time: %.3f ms\n", t_med)
    @printf("Memory:      %.3f MiB\n", b.memory / 1024^2)
    return t_med
end

t_full = run_benchmark("Full Attention (Standard)", full_attn, x)
t_prime = run_benchmark("Prime Attention", prime_attn, x)
t_sq = run_benchmark("Square Attention", sq_attn, x)
t_mc = run_benchmark("Mian-Chowla Attention", mc_attn, x)


@printf "%-25s | %-12s | %s\n" "Model" "Time (ms)" "Speedup vs Full"
@printf "%-25s | %-12.3f | %s\n" "Full Attention" t_full "1.00x (Baseline)"
@printf "%-25s | %-12.3f | %.2fx\n" "Prime Attention" t_prime (t_full / t_prime)
@printf "%-25s | %-12.3f | %.2fx\n" "Square Attention" t_sq (t_full / t_sq)
@printf "%-25s | %-12.3f | %.2fx\n" "Mian-Chowla Attention" t_mc (t_full / t_mc)
