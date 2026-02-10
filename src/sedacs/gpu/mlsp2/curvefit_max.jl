import NonlinearSolve: LevenbergMarquardt, NonlinearFunction
import CurveFit: solve, NonlinearCurveFitProblem
import ForwardDiff
using LinearAlgebra

### Vanilla SP2, inferred for a specific chemical potential

function calc_μ_sp2(b::Vector{Bool})
    prod = 1.0
    y = 0.5
    for i in length(b):-1:1
        if b[i]
            y = sqrt(y)
            prod *= 2y
        else
            y = 1. - sqrt(1. - y)
            prod *= (-2y + 2)
        end
    end
    return (y, 4prod)
end

function build_sp2(μ, nlayers)
    b = Bool[]
    for _ in 1:nlayers
        μ_approx, _ = calc_μ_sp2(b)
        push!(b, μ_approx < μ)
    end
    return b
end

function params_from_sp2(; μ, β, nlayers)
    θ = zeros(4, nlayers)

    # Standard SP2 iterations up to cutoff associated with thermal smearing
    sp2_nlayers = nlayers
    #sp2_nlayers = min(max(round(Int, 4.75log(β)-6.6), 1), nlayers)
    b = build_sp2(μ, sp2_nlayers)

    # Parameters associated with SP2
    for i in 1:sp2_nlayers
        if b[i]
            θ[:, i] = [1, 0, 0, 0] # x' = x^2
        else
            θ[:, i] = [-1, 2, 0, 0] # x' = 2x - x^2
        end
    end

    # Additional layers initialized to "pass through"
    for i in sp2_nlayers+1:nlayers
        θ[:, i] = [0, 1, 0, 0] # x' = x
    end

    return θ
end

### Model definition and Jacobian

β = 54
μ = 0.25
fermi_fn(x; β, μ) = 1 / (1 + exp(β*(x-μ)))

nlayers = 15
θ0 = params_from_sp2(; μ, β, nlayers)[:]

X = collect(range(0, 1, length=1000))
Y = @. 1 - fermi_fn(X; β, μ)

function flatten(theta, d_)
    n = size(theta,1)
    d_index = n * (n + 1) ÷ 2 + n + 1

    theta_vec = zeros(d_index + n)

    for i in range(1,n)
        for j in range(0,i)
            theta_vec[ i * (i + 1) ÷ 2 + j ] = theta[i,j+1]
        end
    end

    theta_vec[d_index:end] = d_

    return theta_vec
end

function recover(theta_vec, n)
    theta = zeros(eltype(theta_vec), n, n+1)

    for i in range(1,n)
        for j in range(0,i)
            theta[i,j+1] = theta_vec[ i * (i + 1) ÷ 2 + j ]
        end
    end

    d_index = n * (n + 1) ÷ 2 + n + 1

    d_ = theta_vec[d_index:end]

    return (theta, d_)
end

function generate_g(n,s)
    function g(theta_vec, x)

        theta, d_ = recover(theta_vec, n)

        y = [ones(eltype(theta), length(x)), eltype(theta).(x)]
        for t in (theta[x,:] for x in 1:size(theta,1))
            push!(y, sum( th * x for (th,x) in zip(t, y) ) .^ 2 )
        end

        return 1 .- sum( d * x for (d,x) in zip(d_,y) ) .- s * y[end]
    end

    return g
end

model = generate_g(nlayers, sign(θ0[1, end]))

function convert_model(W)

    A1, B1, C1, D1  = (W[x,:] for x in 1:size(W,1))
    A0, B0, C0, D0 = (pushfirst!( W[x,1:end-1], [1,0,0,0][x] ) for x in 1:size(W,1))

    A = sqrt.(abs.(A1)) .* sign.(A0)
    B = sqrt.(abs.(A1)) .* ( -B0.^2 ./ 4A0 + C0 + B1 ./ 2A1 )

    D = D1 .* sign.(A0)

    d0 = sum( D1 .* ( C0 - B0.^2 ./ 4A0 ) ) + C1[end] - B1[end]^2 / 4A1[end]

    # convert to full MaxSP2 coefficient matrix
    theta = diagm( 1 => A )[1:end-1,:]
    theta[:,1] = B

    d_ = copy(D)
    d_ = pushfirst!(d_, d0)

    return (A,B,D, theta, d_)
end

A_, B_, D_, theta, d_ = convert_model(reshape(θ0, 4, :))

# println(A_)
# println(B_)
# println(D_)

θ0_ = flatten(theta, d_)

# println(θ0)
# println(typeof(θ0))

# function model(θ, X)
#     θ = reshape(θ, 4, :)

#     layer = 1
#     Z = @. θ[4, layer] * X
#     A = @. θ[1, layer] * X^2 + θ[2, layer] * X + θ[3, layer]

#     for layer in 2:size(θ, 2)
#         Z += @. θ[4, layer] * A
#         A = @. θ[1, layer] * A^2 + θ[2, layer] * A + θ[3, layer]
#     end
#     Z += A
#     return Z
# end

config = ForwardDiff.JacobianConfig(θ -> model(θ, X), θ0_)

function jac(θ, X)
    check = Val{false}()
    ForwardDiff.jacobian(θ -> model(θ, X), θ, config, check)
end

# function jac(θ, X)
#     J = zeros(length(X), length(θ))
#     ESP2.jacobian_inplace(J, θ, X)
#     return J
# end

@time jac(θ0_, X)
@time jac(θ0_, X)

### CurveFit fitting

nonfn = NonlinearFunction(model; jac) # Uses ForwardDiff if jac argument is omitted
prob = NonlinearCurveFitProblem(nonfn, θ0_, X, Y)
# `disable_geodesic=Val(true)` would significantly reduce quality
@time sol_cf = solve(prob, LevenbergMarquardt(); reltol=1e-30, abstol=1e-30, maxiters=10000)
θ_cf = sol_cf.u
Y_cf = model(θ_cf, X)
norm(Y - Y_cf) / sqrt(length(Y))
norm(θ0_ - θ_cf)

@show sol_cf.u

@show norm(Y - Y_cf) / sqrt(length(Y))

### LsqFit fitting, slower by > 10x 

# import LsqFit

# # The LsqFit API takes arguments in reverse order: (pts, params). 
# @time sol_lf = LsqFit.curve_fit((X, θ) -> model(θ, X), (X, θ) -> jac(θ, X), X, Y, θ0; maxIter=1_000_000)
# # sol_lf = LsqFit.curve_fit((X, θ) -> model(θ, X), X, Y, θ0; maxIter=100_000)
# θ_lf = sol_lf.param
# Y_lf = model.(Ref(θ_lf), X)
# norm(Y - Y_lf) / sqrt(length(Y))
# norm(θ0 - θ_lf)

#### Plotting


# using GLMakie

# #=
# fig = Figure(size = (600, 600))
# lines(fig[1, 1], X, model(θ0, X); label="Guess")
# lines!(X, Y_cf; label="Refined")
# lines!(X, Y; color=:red, linestyle=:dash, label="Reference")
# axislegend()
# lines(fig[2, 1], X, model(θ0, X); label="Guess")
# lines!(X, Y_lf; label="Refined")
# lines!(X, Y; color=:red, linestyle=:dash, label="Reference")
# axislegend()
# =#

# lines(X, Y_cf - Y; label="Error CurveFit.jl")
# lines!(X, Y_lf - Y; label="Error LsqFit.jl")
# axislegend()