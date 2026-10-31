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

    # Central finite-difference gradient of `loss()` w.r.t. each entry of `p`
    function finite_diff(loss, p; h = 1.0e-6)
        grad = similar(p)
        for i in eachindex(p)
            p_i = p[i]
            p[i] = p_i + h
            loss_plus = loss()
            p[i] = p_i - h
            loss_minus = loss()
            p[i] = p_i
            grad[i] = (loss_plus - loss_minus) / (2 * h)
        end
        return grad
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

    @testset "Kernel vs Dense Reference" begin
        # Softmax over the unique keys {j <= i : j <= G || i - j <= W || (i - j) in S}
        function dense_reference(q, k, v, S, W, G)
            y = zero(q)
            for i in axes(q, 2)
                attended = filter(j -> j <= G || i - j <= W || (i - j) in S, 1:i)
                scores = [dot(q[:, i], k[:, j]) for j in attended] / sqrt(size(q, 1))
                weights = exp.(scores .- maximum(scores))
                y[:, i] = v[:, attended] * weights / sum(weights)
            end
            return y
        end

        d_head, n = 4, 10
        # The last two (G, W) pairs have G + W >= n: every key is global or in the window
        configs = [(0, 0), (0, 3), (2, 0), (1, 1), (2, 3), (3, 7), (12, 2)]
        for factory in [PrimeSelfAttention, SquareSelfAttention, MianChowlaSelfAttention]
            S = factory(d_head; seq_len_max = n).indices_list
            @testset "$factory, G = $G, W = $W" for (G, W) in configs
                q, k, v = (randn(rng, d_head, n) for _ in 1:3)
                y = similar(q)
                PrimeAttention.sparse_index_kernel!(y, q, k, v, S, W, G)
                @test y ≈ dense_reference(q, k, v, S, W, G)
            end
        end
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

    @testset "Gradient Correctness" begin
        # Compare Enzyme gradients with finite differences, in Float64 for accuracy
        x64 = rand(rng, Float64, dims, seq_len, batch)
        target64 = rand(rng, Float64, dims, seq_len, batch)

        for factory in [PrimeSelfAttention, SquareSelfAttention, MianChowlaSelfAttention]
            layer = factory(dims; seq_len_max = 64)
            ps, st = Lux.setup(rng, layer)
            ps64 = f64(ps)

            dps = Enzyme.make_zero(ps64)
            dx = zero(x64)

            # The input is Duplicated, not Const: for a Const 3-D input, Lux's `Dense`
            # needs runtime activity, which then overwrites the input with its gradient
            Enzyme.autodiff(
                Enzyme.Reverse,
                compute_loss,
                Active,
                Duplicated(ps64, dps),
                Const(layer),
                Duplicated(x64, dx),
                Const(target64),
                Const(st),
            )

            loss = () -> compute_loss(ps64, layer, x64, target64, st)
            @test dps.Wi.weight ≈ finite_diff(loss, ps64.Wi.weight) rtol = 1.0e-5
            @test dps.Wi.bias ≈ finite_diff(loss, ps64.Wi.bias) rtol = 1.0e-5
            @test dps.Wo.weight ≈ finite_diff(loss, ps64.Wo.weight) rtol = 1.0e-5
            @test dps.Wo.bias ≈ finite_diff(loss, ps64.Wo.bias) rtol = 1.0e-5
            @test dx ≈ finite_diff(loss, x64) rtol = 1.0e-5
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
