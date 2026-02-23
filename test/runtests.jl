using PrimeAttention
using Test
using Lux
using Enzyme
using LinearAlgebra
using Random

@testset "PrimeAttention.jl" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    dims = 16
    seq_len = 32
    batch = 2
    x = rand(Float32, dims, seq_len, batch)
    target = rand(Float32, dims, seq_len, batch)

    function compute_loss(ps, layer, x, target, st)
        y, _ = layer(x, ps, st)
        return sum(abs2, y .- target) / length(target)
    end

    @testset "Construction & Forward" begin
        # 1. Prime
        p_layer = PrimeSelfAttention(
            dims;
            heads = 1,
            window = 3,
            global_tokens = 2,
            seq_len_max = 128,
        )
        @test p_layer isa SparseIndexAttention
        @test length(p_layer.indices_list) > 0

        ps_p, st_p = Lux.setup(rng, p_layer)
        y_p, _ = p_layer(x, ps_p, st_p)
        @test size(y_p) == (dims, seq_len, batch)

        # 2. Square
        s_layer = SquareSelfAttention(
            dims;
            heads = 1,
            window = 3,
            global_tokens = 2,
            seq_len_max = 128,
        )
        @test s_layer isa SparseIndexAttention
        @test s_layer.indices_list[1] == 1 # First square is 1

        ps_s, st_s = Lux.setup(rng, s_layer)
        y_s, _ = s_layer(x, ps_s, st_s)
        @test size(y_s) == (dims, seq_len, batch)

        # 3. Mian-Chowla
        m_layer = MianChowlaSelfAttention(
            dims;
            heads = 1,
            window = 3,
            global_tokens = 2,
            seq_len_max = 128,
        )
        @test m_layer isa SparseIndexAttention
        @test m_layer.indices_list[1] == 1 # First MC number is 1

        ps_m, st_m = Lux.setup(rng, m_layer)
        y_m, _ = m_layer(x, ps_m, st_m)
        @test size(y_m) == (dims, seq_len, batch)
    end

    @testset "Gradient Pass" begin
        # Test gradient through the universal kernel for different index sets
        for factory in [PrimeSelfAttention, SquareSelfAttention, MianChowlaSelfAttention]
            layer = factory(dims; seq_len_max = 64)
            ps, st = Lux.setup(rng, layer)

            dps = Enzyme.make_zero(ps)

            Enzyme.autodiff(
                set_runtime_activity(Enzyme.Reverse),
                compute_loss,
                Active,
                Duplicated(ps, dps),
                Const(layer),
                Const(x),
                Const(target),
                Const(st),
            )

            # Check gradients for input and output projection weights
            @test norm(dps.Wi.weight) > 0
            @test norm(dps.Wo.weight) > 0
        end
    end

    @testset "Index Check" begin
        # Verify indices are actually different
        p_indices = PrimeSelfAttention(dims; seq_len_max = 50).indices_list
        s_indices = SquareSelfAttention(dims; seq_len_max = 50).indices_list

        @test p_indices != s_indices
        @test 2 in p_indices # 2 is prime
        @test !(2 in s_indices) # 2 is not square
        @test 4 in s_indices # 4 is square
    end
end
