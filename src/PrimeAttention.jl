module PrimeAttention

using Flux
using Zygote
using Primes
using LinearAlgebra
using Statistics

# Include sub-files
include("kernel.jl")
include("layer.jl")

# Export the user-facing layer
export PrimeSelfAttention

end # module