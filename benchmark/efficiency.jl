# Run from the repository root with:
#   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#   julia --project=benchmark benchmark/efficiency.jl
using PrimeAttention
using Lux
using BenchmarkTools
using Printf
using Random

include("full_attention.jl")

# Configuration
const DIMS = 64      # Embedding dimension
const SEQ_LEN = 2048 # Sequence length (N)
const BATCH_SIZE = 4 # Batch size
const GLOBAL = 3     # Global tokens
const WINDOW = 3     # Local window size
const HEADS = 1      # Attention heads

@printf("Config: Sequence Length=%d, Batch=%d, Dims=%d, Heads=%d\n", SEQ_LEN, BATCH_SIZE, DIMS, HEADS)

rng = Random.default_rng()
Random.seed!(rng, 0)

# Dummy Input
x = randn(rng, Float32, DIMS, SEQ_LEN, BATCH_SIZE)

# Initialize Models
full_attn = StandardFullAttention(DIMS; heads = HEADS)

prime_attn = PrimeSelfAttention(
    DIMS;
    heads = HEADS,
    window = WINDOW,
    global_tokens = GLOBAL,
    seq_len_max = SEQ_LEN,
)

sq_attn = SquareSelfAttention(
    DIMS;
    heads = HEADS,
    window = WINDOW,
    global_tokens = GLOBAL,
    seq_len_max = SEQ_LEN,
)

mc_attn = MianChowlaSelfAttention(
    DIMS;
    heads = HEADS,
    window = WINDOW,
    global_tokens = GLOBAL,
    seq_len_max = SEQ_LEN,
)

function run_benchmark(name, model, input, rng)
    println("Benchmarking $name")
    ps, st = Lux.setup(rng, model)
    model(input, ps, st)
    b = @benchmark $model($input, $ps, $st)

    t_med = median(b.times) / 1.0e6

    @printf("Median Time: %.3f ms\n", t_med)
    @printf("Memory:      %.3f MiB\n", b.memory / 1024^2)
    return t_med
end

t_full = run_benchmark("Full Attention (Standard)", full_attn, x, rng)
t_prime = run_benchmark("Prime Attention", prime_attn, x, rng)
t_sq = run_benchmark("Square Attention", sq_attn, x, rng)
t_mc = run_benchmark("Mian-Chowla Attention", mc_attn, x, rng)


@printf "%-25s | %-12s | %s\n" "Model" "Time (ms)" "Speedup vs Full"
@printf "%-25s | %-12.3f | %s\n" "Full Attention" t_full "1.00x (Baseline)"
@printf "%-25s | %-12.3f | %.2fx\n" "Prime Attention" t_prime (t_full / t_prime)
@printf "%-25s | %-12.3f | %.2fx\n" "Square Attention" t_sq (t_full / t_sq)
@printf "%-25s | %-12.3f | %.2fx\n" "Mian-Chowla Attention" t_mc (t_full / t_mc)
