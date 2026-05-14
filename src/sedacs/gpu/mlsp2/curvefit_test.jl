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
    sp2_nlayers = min(max(round(Int, 4.75log(β)-6.6), 1), nlayers)
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

β = 1500
μ = 1/3

freeze = false

fermi_fn(x; β, μ) = 1 / (1 + exp(β*(x-μ)))

nlayers = 26
θ0 = params_from_sp2(; μ, β, nlayers)[:]

# @show θ0

X = collect(range(0, 1, length=250))
Z = μ .+ collect(range(-1,1, length=250)) ./ .2β
# Z = collect(range(0, 1, length=250))
# Z = @. 2atanh(2Z-1.)/β + μ
Z = clamp.( Z, 0., 1. )
X = vcat(X, Z)

Y = @. 1 - fermi_fn(X; β, μ)

if freeze

    if θ0[4nlayers - 3] == 1 && θ0[4nlayers-7] == -1

        function model(θ, X)
            θ = reshape(θ, 4, :)

            layer = 1
            Z = @. θ[4, layer] * X
            A = @. θ[1, layer] * X^2 + θ[2, layer] * X + θ[3, layer]

            for layer in 2:size(θ, 2) - 2
                Z += @. θ[4, layer] * A
                A = @. θ[1, layer] * A^2 + θ[2, layer] * A + θ[3, layer]
            end
            Z += @. (2A-A^2)^2
            return Z
        end
    
    elseif θ0[4nlayers - 3] == -1 && θ0[4nlayers-7] == 1

        function model(θ, X)
            θ = reshape(θ, 4, :)

            layer = 1
            Z = @. θ[4, layer] * X
            A = @. θ[1, layer] * X^2 + θ[2, layer] * X + θ[3, layer]

            for layer in 2:size(θ, 2) - 2
                Z += @. θ[4, layer] * A
                A = @. θ[1, layer] * A^2 + θ[2, layer] * A + θ[3, layer]
            end
            Z += @. 2A^2 - A^4
            return Z
        end

    else
        throw("freeze option chosen but last two layers do not alternate. Choose a different μ or layer count.")
    end
    
else

    function model(θ, X)
        θ = reshape(θ, 4, :)

        layer = 1
        Z = @. θ[4, layer] * X
        A = @. θ[1, layer] * X^2 + θ[2, layer] * X + θ[3, layer]

        for layer in 2:size(θ, 2)
            Z += @. θ[4, layer] * A
            A = @. θ[1, layer] * A^2 + θ[2, layer] * A + θ[3, layer]
        end
        Z += A
        return Z
    end

end

config = ForwardDiff.JacobianConfig(θ -> model(θ, X), θ0)

function jac(θ, X)
    check = Val{false}()
    ForwardDiff.jacobian(θ -> model(θ, X), θ, config, check)
end

# function jac(θ, X)
#     J = zeros(length(X), length(θ))
#     ESP2.jacobian_inplace(J, θ, X)
#     return J
# end

@time jac(θ0, X)

### CurveFit fitting

nonfn = NonlinearFunction(model; jac) # Uses ForwardDiff if jac argument is omitted
prob = NonlinearCurveFitProblem(nonfn, θ0, X, Y)
# `disable_geodesic=Val(true)` would significantly reduce quality
@time sol_cf = solve(prob, LevenbergMarquardt(); reltol=1e-30, abstol=1e-30, maxiters=10000)
θ_cf = sol_cf.u
Y_cf = model.(Ref(θ_cf), X)
norm(Y - Y_cf) / sqrt(length(Y))
norm(θ0 - θ_cf)

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