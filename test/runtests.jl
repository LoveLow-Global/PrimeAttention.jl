using PrimeAttention
using Test
using Flux
using Zygote
using LinearAlgebra

@testset "PrimeAttention.jl" begin
    
    @testset "Construction" begin
        # Args: dims, heads, window, global
        layer = PrimeSelfAttention(16, 1, 3, 2)
        @test layer isa PrimeSelfAttention
        @test length(layer.primes_list) > 0
        @test layer.window == 3
        @test layer.global_tokens == 2
    end

    @testset "Forward Pass" begin
        seq_len = 50
        dim = 16
        batch = 2

        layer = PrimeSelfAttention(dim, 1, 3, 2; seq_len_max = 128)
        
        x = rand(Float32, dim, seq_len, batch)
        y = layer(x)
        
        @test size(y) == (dim, seq_len, batch)
        @test !any(isnan, y)
    end

    @testset "Gradient Pass" begin
        seq_len = 20
        dim = 8

        # FIX: Removed the extra '128' positional argument
        layer = PrimeSelfAttention(dim, 1, 3, 2; seq_len_max = 128)
        
        x = rand(Float32, dim, seq_len, 1)
        target = rand(Float32, dim, seq_len, 1)
        
        loss(m, x, y) = Flux.mse(m(x), y)
        
        grads = Flux.gradient(m -> loss(m, x, target), layer)
        
        @test grads[1] !== nothing
        @test norm(grads[1][:Wi].weight) > 0
    end

end