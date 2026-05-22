import NonlinearSolve: LevenbergMarquardt, NonlinearFunction
import CurveFit: solve, NonlinearCurveFitProblem
import ForwardDiff
using LinearAlgebra

# perturbation strength (bifurcation at 0.41421)
l = 0.5

# number of final pure SP2 smoothing layers
k = 6

# Model definition
mu = 1/3
nlayers = 30

# USER SET: f_0 quadratic coefficients
a, b, c = 1+l, -l, 0

@assert a > 0 # make sure f_0 is the concave-up one (see f0_inv).

# NOTE: f_1 is defined as the mirror of f_0, i.e. f_1(x) = 1-f_0(1-x)

# quadratic formula to solve ax^2 + bx + c = y
# NOTE: will only return solutions to the sign(a) of vertex!
# i.e. if b<0 and a>0, x > -b/2a.
function f0_inv(y)
    return (sqrt(b^2 - 4a*(c-y)) - b)/2a
end

function f0_prime(x)
    return 2a*x+b 
end

# compute choice of f_0 vs f_1 sequence for SP2 at a specific chemical potential

# alternative method which computes mu_i as the location where f(mu_i) = 1/2.
function calc_mu_sp2(s::Vector{Bool}, n_k::Int)
    prod = 1.0
    y = 0.5
    for i in length(s):-1:1
        if i > n_k
            if s[i]
                y = sqrt(y)             # inverse of x^2
                prod *= 2y              # derivative of x^2
            else
                y = 1-sqrt(1-y)         # inverse of 2x - x^2
                prod *= 2 - 2y          # derivative of 2x - x^2
            end
        else
            if s[i]
                y = f0_inv(y)           # inverse of f_0
                prod *= f0_prime(y)     # derivative of f_0
            else
                y = 1-f0_inv(1-y)       # inverse of f_1
                prod *= f0_prime(1-y)   # derivative of f_1
            end
        end
    end
    return (y, 4prod)               # return the location of mu_i and f'(mu_i)
end

# choose each next layer based on whether mu_i needs to be pulled up or down
function build_sp2(mu, nlayers, k)
    s = Bool[]
    beta_eff = 0
    for _ in 1:nlayers
        mu_i, beta_eff = calc_mu_sp2(s, nlayers-k)
        push!(s, mu_i < mu)
    end
    return s, beta_eff
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
function params_from_sp2(; mu, nlayers, k)
    theta = zeros(4, nlayers)

    s, beta_eff = build_sp2(mu, nlayers, k)

    # Parameters associated with SP2
    for i in 1:nlayers
        if i > nlayers - k
            if s[i]
                theta[:, i] = [1, 0, 0, 0]              # x' = x^2
            else
                theta[:, i] = [-1, 2, 0, 0]             # x' = 2x - x^2
            end
        else
            if s[i]
                theta[:, i] = [a, b, c, 0]              # x' = f_0(x)
            else
                theta[:, i] = [-a, 2a+b, 1-a-b-c, 0]    # x' = f_1(x)
            end
        end
    end

    return theta, beta_eff
end

# initialize model weights to the embedding of the equivalent truncated SP2 method
theta0, beta = params_from_sp2(; mu, nlayers, k)[:]

# function we try to match
fermi_fn(x; beta, mu) = 1 / (1 + exp(beta*(x-mu)))

# sample points (uniformly sampled across [0,1] and also focused on fermi level)
X = collect(range(0, 1, length=250))
Z = mu .+ collect(range(-1,1, length=250)) ./ .2beta
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

@show initial_err = maximum(abs.(Y - model(theta0, X)))

function jac(theta, X)
    check = Val{false}()
    ForwardDiff.jacobian(theta -> model(theta, X), theta, config, check)
end

@time jac(theta0, X)

# set up as a nonlinear curve-fitting problem and solve for optimal coefficients
nonfn = NonlinearFunction(model; jac) # Uses ForwardDiff if jac argument is omitted
prob = NonlinearCurveFitProblem(nonfn, theta0, X, Y)
# `disable_geodesic=Val(true)` would significantly reduce quality
@time sol_cf = solve(prob, LevenbergMarquardt(); reltol=1e-30, abstol=1e-30, maxiters=10000)
@show sol_cf.u

# print least-squares error (what the solver has minimized)
Y_cf = model.(Ref(sol_cf.u), X)
@show norm(Y - Y_cf)

# show worst-case error (higher sample rate than training)

X = collect(range(0, 1, length=5000))
Z = mu .+ collect(range(-1,1, length=5000)) ./ .2beta
Z = clamp.( Z, 0., 1. )
X = vcat(X, Z)

Y_cf = model.(Ref(sol_cf.u), X)
Y = @. fermi_fn(X; beta, mu)

# print actual worst case error
@show maximum(abs.(Y - Y_cf))

### LsqFit fitting, slower by > 10x 

# import LsqFit

# # The LsqFit API takes arguments in reverse order: (pts, params). 
# @time sol_lf = LsqFit.curve_fit((X, theta) -> model(theta, X), (X, theta) -> jac(theta, X), X, Y, theta0; maxIter=1_000_000)
# # sol_lf = LsqFit.curve_fit((X, theta) -> model(theta, X), X, Y, theta0; maxIter=100_000)
# theta_lf = sol_lf.param