using LinearAlgebra
using Printf
using LsqFit


### Vanilla SP2, targeting specific chemical potential

function calc_μ_sp2(b::Vector{Bool})
    prod = 1.0
    y = 0.5
    for i = length(b):-1:1
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
    for i = 1:nlayers
        μ_approx, _ = calc_μ_sp2(b)
        push!(b, μ_approx < μ)
    end
    return b
end

function eval_sp2(b::Vector{Bool}, x)
    y = x
    for i = 1:length(b)
        if b[i]
            y = y^2
        else
            y = 2y - y^2
        end
    end
    return y
end


const layer_width = 4

function params_from_sp2(μ, nlayers)
    θ = zeros(layer_width, nlayers)
    b = build_sp2(μ, nlayers)

    for i = 1:nlayers
        if b[i]
            θ[:, i] = [1.,  0., 0., 0.] # x' = x^2
        else
            θ[:, i] = [-1., 2., 0., 0.] # x' = 2x - x^2
        end
    end

    return reshape(θ, :)
end


### Generalized model

# Postprocessing for final model output, and derivative
fermi_transf_1(Y) = 1 - Y
fermi_transf_2(Y) = -1.
entropy_transf_1(Y) = 4log(2) * (Y - Y^2)
entropy_transf_2(Y) = 4log(2) * (1 - 2Y)

function model_inplace(res, x, θ, f)
    npts = length(x)
    θ = reshape(θ, layer_width, :)
    nlayers = size(θ, 2)

    fill!(res, zero(eltype(res)))

    for j = 1:npts
        y = x[j]
        Y = zero(eltype(res))
        for i = 1:nlayers
            Y += θ[4, i] * y
            y = θ[1, i] * y^2 + θ[2, i] * y + θ[3, i]
        end
        Y += y
        res[j] = f(Y)
    end
end

model_inplace_fermi(res, x, θ)   = model_inplace(res, x, θ, fermi_transf_1)
model_inplace_entropy(res, x, θ) = model_inplace(res, x, θ, entropy_transf_1)

function model(x, θ, f)
    res = Vector{eltype(x)}(undef, length(x))
    model_inplace(res, x, θ, f)
    return res
end

model_fermi(x, θ)   = model(x, θ, fermi_transf_1)
model_entropy(x, θ) = model(x, θ, entropy_transf_1)


function jacobian_inplace(J::Array{Float64,2}, x, θ, df_dY)
    npts = length(x)
    θ = reshape(θ, layer_width, :)
    nlayers = size(θ, 2)

    J = reshape(J, npts, layer_width, nlayers)
    y = zeros(eltype(x), nlayers+1)

    for j = 1:npts

        # forward calculation
        y[1] = x[j]
        Y = zero(eltype(J))
        for i = 1:nlayers
            Y += θ[4, i] * y[i]
            y[i+1] = θ[1, i] * y[i]^2 + θ[2, i] * y[i] + θ[3, i]
        end
        Y += y[nlayers+1]
        α = df_dY(Y)

        # backward calculation
        z = 1 # z_{n+1}
        for i = nlayers:-1:1
            # z = z_{i+1}
            J[j, 1, i] = α * z * y[i]^2
            J[j, 2, i] = α * z * y[i]
            J[j, 3, i] = α * z
            J[j, 4, i] = α * y[i]

            z = θ[4, i] + z * (2θ[1, i] * y[i] + θ[2, i])
        end
    end
end

jacobian_inplace_fermi(J, x, θ)   = jacobian_inplace(J, x, θ, fermi_transf_2)
jacobian_inplace_entropy(J, x, θ) = jacobian_inplace(J, x, θ, entropy_transf_2)

#= 
function test_jacobian(x, θ, f, df_dY, idx)
    θp = copy(θ)
    θn = copy(θ)

    ϵ = 1e-6
    θp[idx] += ϵ
    θn[idx] -= ϵ
    J_idx = (model(x, θp, f) - model(x, θn, f)) ./ 2ϵ

    J = zeros(length(x), length(θ))
    jacobian_inplace(J, x, θ, df_dY)

    norm(J[:, idx] - J_idx)
end
test_jacobian(x, θ_fermi, fermi_transf_1,   fermi_transf_2,   22)
test_jacobian(x, θ_sp2,   entropy_transf_1, entropy_transf_2, 22)
=#


### Sample points

function sample_by_density(x0, xlo, xhi, density)
    pts = [x0]

    # add points below
    dx = 1/density(x0)
    x = x0-dx
    while x > xlo
        push!(pts, x)
        dx = 2/density(x) - dx
        x -= dx
    end
    push!(pts, xlo)

    reverse!(pts)

    # add points above
    dx = 1/density(x0)
    x = x0+dx
    while x < xhi
        push!(pts, x)
        dx = 2/density(x) - dx
        x += dx
    end
    push!(pts, xhi)

    return pts
end

function sample_weights(pts)
    @assert length(pts) ≥ 2
    weights = zeros(eltype(pts), length(pts))
    # endpoints get half the normal weight
    weights[1]       = (pts[2] - pts[1]) / 2
    weights[end]     = (pts[end] - pts[end-1]) / 2
    # interior points collect half the weight from each of the two neighbor intervals
    weights[2:end-1] = (pts[3:end] - pts[1:end-2]) ./ 2
    return weights
end


### Model training routines

fermi_fn(x, β, μ)   = 1 / (1 + exp(β*(x-μ)))

function energy_fn(x, β, μ)
    a = β*(x-μ)
    if a > 0
        - β^-1 * log(1 + exp(-a))
    else
        # avoid overflow for very negative a
        - β^-1 * (log(1 + exp(a)) - a)
    end
end

entropy_fn(x, β, μ) = - β * (energy_fn(x, β, μ) - fermi_fn(x, β, μ) * (x - μ))

function generate_model(; β, μ, max_iter, npts_scale=1.0, nlayers=round(Int64, 4.75log(β)-6.6))

    # Sample points more densely near x=μ
    npts = npts_scale * 80log(β)
    w = sqrt(β)
    sample_density(x) = (npts/2) + (npts/2) * (w/2) * sech(w * (x - μ))^2
    x = sample_by_density(μ, 0, 1, sample_density)
    weight = sample_weights(x)

    # Initialize model with SP2
    θ_sp2 = params_from_sp2(μ, nlayers)

    # show_trace = true
    @time fit_fermi   = curve_fit(model_inplace_fermi,
                                  jacobian_inplace_fermi,
                                  x,
                                  fermi_fn.(x, β, μ),
                                  θ_sp2;
                                  maxIter=max_iter,
                                  inplace=true)
    @time fit_entropy = curve_fit(model_inplace_entropy,
                                  jacobian_inplace_entropy,
                                  x,
                                  entropy_fn.(x, β, μ),
                                  θ_sp2;
                                  maxIter=max_iter,
                                  inplace=true)

    return (; θ_sp2, θ_fermi=coef(fit_fermi), θ_entropy=coef(fit_entropy), x)
end


using JLSO

function read_or_generate_models(filename, overwrite=false)    
    need_calculations = overwrite || !isfile(filename)

    if need_calculations
        max_iters = [1_000, 4_000, 16_000, 64_000]
        βs = [40, 400, 4000]
        μs = [0.1, 0.5]

        all_models = map(Iterators.product(βs, μs, max_iters)) do (β, μ, max_iter)
            println("Generating β=$β, μ=$μ, max_iter=$max_iter.")
            (θ_sp2, θ_fermi, θ_entropy, x) = generate_model(; β, μ, max_iter, npts_scale=1)
            (; β, μ, max_iter, θ_sp2, θ_fermi, θ_entropy, x)
        end
        all_models = reshape(all_models, :)

        JLSO.save(filename, :models => all_models)
    else
        all_models = JLSO.load(filename)[:models]
    end

    return all_models
end


### Figuring out width of SP2 -- β vs nlayers

# Empirical fitting function:
# 4.75log(β) + 3.2log(sin(π*μ)) - 6.6

#= 
# μ = 1/2
[40 11; 400 22; 4000 33; 40_000 44]

# μ = 1/10
[40 8; 400 19; 4000 29; 40_000 40]

# μ = 1/100
[40 4; 400 11; 4000 22; 40_000 33]

# μ = 1/1000
[40 ?; 400 ?; 4000 ?; 40_000 25]
=#