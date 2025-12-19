using PrimeAttention
using Test
using Flux
using Zygote
using LinearAlgebra

@testset "PrimeAttention.jl" begin
    
    dims = 16
    seq_len = 32
    batch = 2
    x = rand(Float32, dims, seq_len, batch)
    target = rand(Float32, dims, seq_len, batch)
    loss(m, x, t) = Flux.mse(m(x), t)

    @testset "Construction & Forward" begin
        # 1. Prime
        p_layer = PrimeSelfAttention(dims; heads=1, window=3, global_tokens=2, seq_len_max=128)
        @test p_layer isa SparseIndexAttention
        @test length(p_layer.indices_list) > 0
        @test size(p_layer(x)) == (dims, seq_len, batch)
        
        # 2. Square
        s_layer = SquareSelfAttention(dims; heads=1, window=3, global_tokens=2, seq_len_max=128)
        @test s_layer isa SparseIndexAttention
        @test s_layer.indices_list[1] == 1 # First square is 1
        @test size(s_layer(x)) == (dims, seq_len, batch)

        # 3. Mian-Chowla
        m_layer = MianChowlaSelfAttention(dims; heads=1, window=3, global_tokens=2, seq_len_max=128)
        @test m_layer isa SparseIndexAttention
        @test m_layer.indices_list[1] == 1 # First MC number is 1
        @test size(m_layer(x)) == (dims, seq_len, batch)
    end

    @testset "Gradient Pass" begin
        # Test gradient through the universal kernel for different index sets
        for factory in [PrimeSelfAttention, SquareSelfAttention, MianChowlaSelfAttention]
            layer = factory(dims; seq_len_max=64)
            grads = Flux.gradient(m -> loss(m, x, target), layer)
            
            @test grads[1] !== nothing
            # Check gradients for input projection weights
            @test norm(grads[1].Wi.weight) > 0
        end
    end

    @testset "Index Specificity" begin
        # Verify indices are actually different
        p_indices = PrimeSelfAttention(dims; seq_len_max=50).indices_list
        s_indices = SquareSelfAttention(dims; seq_len_max=50).indices_list
        
        @test p_indices != s_indices
        @test 2 in p_indices # 2 is prime
        @test !(2 in s_indices) # 2 is not square
        @test 4 in s_indices # 4 is square
    end
end