module PrimeAttention

using Lux
using Primes

# Include sub-files
include("kernel.jl")
include("layer.jl")

# Export the generic layer and the specific constructors
export SparseIndexAttention,
    PrimeSelfAttention, SquareSelfAttention, MianChowlaSelfAttention

end # module
