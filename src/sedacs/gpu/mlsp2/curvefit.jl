import NonlinearSolve: LevenbergMarquardt, NonlinearFunction, QRFactorization
import CurveFit: solve, NonlinearCurveFitProblem
import ForwardDiff
using LinearAlgebra

# compute choice of f_0 vs f_1 sequence for SP2 at a specific chemical potential

# alternative method which computes mu_i as the location where f(mu_i) = 1/2.
function calc_mu_sp2(b::Vector{Bool})
    prod = 1.0
    y = 0.5
    for i in length(b):-1:1
        if b[i]
            y = sqrt(y)             # inverse of f_0
            prod *= 2y              # derivative of f_0
        else
            y = 1. - sqrt(1. - y)   # inverse of f_1
            prod *= (-2y + 2)       # derivative of f_1
        end
    end
    return (y, 4prod)               # return the location of mu_i and f'(mu_i)
end

# choose each next layer based on whether mu_i needs to be pulled up or down
function build_sp2(mu, nlayers)
    b = Bool[]
    for _ in 1:nlayers
        mu_i, _ = calc_mu_sp2(b)
        push!(b, mu_i < mu)
    end
    return b
end

## version using Eq. 12
# function build_sp2(mu, nlayers)
#     b = Bool[]
#     mu_i = mu
#     for _ in 1:nlayers
#         if abs(mu_i^2 - mu) < abs(2mu_i - mu_i^2 - mu)
#             mu_i = mu_i^2
#             push!(b, true)
#         else
#             mu_i = 2mu_i - mu_i^2 
#             push!(b, false)
#         end
#     end
#     return b
# end

# embed SP2 in MLSP2 weights (theta)
function params_from_sp2(; mu, beta, nlayers)
    theta = zeros(4, nlayers)

    b = build_sp2(mu, nlayers)

    mu_eff, beta_eff = calc_mu_sp2(b)

    # rescaling via Eq. 46
    scale = beta/beta_eff
    shift = mu - mu_eff * scale

    # region of validity of SP2 model via Eq. 48 
    if mu_eff/mu < scale || (mu_eff-1)/(mu-1) < scale
        println("warning: not enough layers!")
        @show beta_eff beta (mu_eff/mu) (mu_eff-1)/(mu-1) scale
        if mu_eff < mu
            # shift mu_eff up to mu, preserve 1
            scale = (mu - 1)/(mu_eff - 1)   # <1
            shift = 1 - scale               # >0
        else
            # shift mu_eff down to mu, preserve 0
            scale = mu/mu_eff
            shift = 0
        end
    end

    # Parameters associated with SP2
    for i in 1:nlayers
        if b[i]
            theta[:, i] = [1, 0, 0, 0]  # x' = x^2
        else
            theta[:, i] = [-1, 2, 0, 0] # x' = 2x - x^2
        end
    end

    # integrate the shift + scale into the first layer:
    # a1,b1,c1,_ = theta[:,1]

    # theta[1,1] = a1 * scale^2
    # theta[2,1] = 2  * scale * shift * a1 + b1 * scale
    # theta[3,1] = a1 * shift^2 + b1 * shift + c1

    return theta
end

# Boltzmann constant [eV/K], temperature [K], spectral width [eV]
kB = 8.617333262e-5
T = 250
eV = 100

# Model definition
#@show beta = eV/(kB * T) * 3/2
beta = 277
mu = 1/3
nlayers = 20

# function we try to match
fermi_fn(x; beta, mu) = 1 / (1 + exp(beta*(x-mu)))

# initialize model weights to the embedding of the equivalent truncated SP2 method
theta0 = params_from_sp2(; mu, beta, nlayers)[:]

# sample points (uniformly sampled across [0,1] and also focused on fermi level)
X = collect(range(0, 1, length=250))
Z = mu .+ X .* 0.15 .- 0.07 #collect(range(-1,1, length=500)) ./ .2beta
Z = clamp.( Z, 0., 1. )
X = vcat(X, Z)

# trial data
Y = @. fermi_fn(X; beta, mu)

# model definition
function model(theta, X)
    theta = reshape(theta, 4, :)

    A = 0 * X 
    for layer in 1:size(theta, 2)
        A += @. theta[4, layer] * X
        X  = @. theta[1, layer] * X^2 + theta[2, layer] * X + theta[3, layer]
    end
    A += X
    return 1 .- A
end

# configure Jacobian evaluation of model using Enzyme.jl
config = ForwardDiff.JacobianConfig(theta -> model(theta, X), theta0)

function jac(theta, X)
    check = Val{false}()
    ForwardDiff.jacobian(theta -> model(theta, X), theta, config, check)
end

@time jac(theta0, X)

# set up as a nonlinear curve-fitting problem and solve for optimal coefficients
nonfn = NonlinearFunction(model; jac) # Uses ForwardDiff if jac argument is omitted
prob = NonlinearCurveFitProblem(nonfn, theta0, X, Y)
# `disable_geodesic=Val(true)` would significantly reduce quality
alg = LevenbergMarquardt(linsolve = QRFactorization())
@time sol_cf = solve(prob, alg; reltol=1e-30, abstol=1e-30, maxiters=10000)
@show sol_cf.u

# print least-squares error (what the solver has minimized)
Y_cf = model.(Ref(sol_cf.u), X)
@show norm(Y - Y_cf)

# show worst-case error (higher sample rate than training)

X_test = collect(range(0, 1, length=5000))
Z_test = mu .+ collect(range(-1,1, length=5000)) ./ .2beta
Z_test = clamp.( Z_test, 0., 1. )
X_test = vcat(X_test, Z_test)

Y_cf = model.(Ref(sol_cf.u), X_test)
Y_test = @. fermi_fn(X_test; beta, mu)

# print actual worst case error
@show maximum(abs.(Y_test - Y_cf))

### ENTROPY

#sol_cf.u[3:4:end] .= 0.0

theta1 = sol_cf.u
a,b,c = theta1[1:3]

alpha = 0.842704
m = mu * (1 - alpha)

# theta1[1] = a * alpha^2
# theta1[2] = 2a * alpha * m + b * alpha
# theta1[3] = a * m^2 + b * m + c

@show theta1

# ignore d, should be minimal.

Y = @. - Y * log(Y) - (1-Y) * log(1-Y)
Y = replace(Y, NaN => 0.0)

# model definition
function modelS(theta, X)
    theta = reshape(theta, 4, :)

    X = alpha * X .+ m

    A = 0 * X 
    for layer in 1:size(theta, 2)
        A += @. theta[4, layer] * X
        X  = @. theta[1, layer] * X^2 + theta[2, layer] * X + theta[3, layer]
    end
    A += X
    return @. 4.0 * log(2.0) * A * (1.0 .- A)
end

# configure Jacobian evaluation of model using Enzyme.jl
configS = ForwardDiff.JacobianConfig(thetax -> modelS(thetax, X), theta1)

function jacS(theta, X)
    check = Val{false}()
    ForwardDiff.jacobian(thetax -> modelS(thetax, X), theta, configS, check)
end

@time jacS(theta1, X)

Y_cf = modelS.(Ref(theta1), X)
@show norm(Y - Y_cf)

using Plots

p = plot(X, [Y Y_cf], label=["reference", "model initialization"])

display(p)

# set up as a nonlinear curve-fitting problem and solve for optimal coefficients
nonfn = NonlinearFunction(modelS; jac) # Uses ForwardDiff if jac argument is omitted
prob = NonlinearCurveFitProblem(nonfn, theta1, X, Y)
# `disable_geodesic=Val(true)` would significantly reduce quality
@time sol_cf = solve(prob, alg; reltol=1e-30, abstol=1e-30, maxiters=10000)
@show sol_cf.u

# print least-squares error (what the solver has minimized)
Y_cf = modelS.(Ref(sol_cf.u), X)
@show norm(Y - Y_cf)

# show worst-case error (higher sample rate than training)

X_test = collect(range(0, 1, length=5000))
Z_test = mu .+ collect(range(-1,1, length=5000)) ./ .2beta
Z_test = clamp.( Z_test, 0., 1. )
X_test = vcat(X_test, Z_test)

Y_cf = modelS.(Ref(sol_cf.u), X_test)
Y_test = @. fermi_fn(X_test; beta, mu)

Y_test = @. -Y_test * log(Y_test) - (1 - Y_test) * log(1 - Y_test)
Y_test = replace(Y_test, NaN => 0.0)

# print actual worst case error
@show maximum(abs.(Y_test - Y_cf))

### LsqFit fitting, slower by > 10x 

# import LsqFit

# # The LsqFit API takes arguments in reverse order: (pts, params). 
# @time sol_lf = LsqFit.curve_fit((X, theta) -> model(theta, X), (X, theta) -> jac(theta, X), X, Y, theta0; maxIter=1_000_000)
# # sol_lf = LsqFit.curve_fit((X, theta) -> model(theta, X), X, Y, theta0; maxIter=100_000)
# theta_lf = sol_lf.param