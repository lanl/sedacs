import NonlinearSolve: LevenbergMarquardt, NonlinearFunction
import CurveFit: solve, NonlinearCurveFitProblem
import ForwardDiff
using LinearAlgebra
import PolynomialRoots

# compute choice of f_0 vs f_1 sequence for SP2 at a specific chemical potential

# alternative method which computes mu_i as the location where f(mu_i) = 1/2.
function calc_mu_sp2(b::Vector{Bool})
    prod = 1.0
    y = 0.5
    for i in length(b):-1:1
        if b[i]
            # solve -3x^4 + 4x^3 = y
            roots = PolynomialRoots.roots([-y, 0, 0, 4, -3])

            # filter to real roots between 0 and 1
            y = real( filter( x -> 0 < real(x) < 1 && isreal(x), roots ) )[1]

            prod *= 12(y^2-y^3)     # derivative of f_0
        else
            # solve 3x^4 - 8x^3 + 6x^2 = y
            roots = PolynomialRoots.roots([-y, 0, 6, -8, 3])

            # filter to real roots between 0 and 1
            y = real( filter( x -> 0 < real(x) < 1 && isreal(x), roots ) )[1]
            
            prod *= 12(y^3-2y^2+y)  # derivative of f_1
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

    # Standard SP2 iterations up to cutoff wherin slope beta/4 is matched
    sp2_nlayers = min(max(round(Int, 4.72log(beta)-6.54), 1), nlayers)
    b = build_sp2(mu, sp2_nlayers)

    # Parameters associated with SP2
    for i in 1:sp2_nlayers
        if b[i]
            theta[:, i] = [1, 0, 0, 0]  # x' = x^2
        else
            theta[:, i] = [-1, 2, 0, 0] # x' = 2x - x^2
        end
    end

    # Additional layers initialized to "pass through"
    for i in sp2_nlayers+1:nlayers
        theta[:, i] = [0, 1, 0, 0]      # x' = x
    end

    return theta
end

# Model definition
beta = 1500
mu = 1/3
nlayers = 26

# function we try to match
fermi_fn(x; beta, mu) = 1 / (1 + exp(beta*(x-mu)))

# initialize model weights to the embedding of the equivalent truncated SP2 method
theta0 = params_from_sp2(; mu, beta, nlayers)[:]

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