### All commands in this file are intended to be run interactively

using GenSP2
using Printf
using Plots
using LinearAlgebra

all_models = read_or_generate_models("/Users/kbarros/Desktop/models.jlso")

β = 400
μ = 0.5
models = filter(m -> (m.β == β) && (m.μ == μ), all_models)
# drop first model
models = models[2:end]

# This is the function that curve_fit() implicitly optimizes
rmse(x, err) = sqrt(sum(sample_weights(x) .* err .^ 2))

begin
    m = models[end]

    f_ref = fermi_fn.(m.x, β, μ)
    g_ref = energy_fn.(m.x, β, μ)
    f_opt = model_fermi(m.x, m.θ_fermi)
    σ_opt = model_entropy(m.x, m.θ_entropy)
    g_opt = @. f_opt * (m.x - μ) - β^-1 * σ_opt

    npts = length(m.x)

    println("# beta = 40, mu = 1/2")
    println("# f(x) = 1 / (1 + exp(beta (x - mu))) ")
    println("# E(x) = - beta^-1 log(1 + exp(-(beta (x-μ))))")
    println("# s(x) = - beta (E(x) - f(x) (x - μ))")
    println("# DATA FORMAT: ")
    println("# x f(x) E(x) f_model(x) s_model(x) E_model(x)")
    for i in 1:npts
        println("$(m.x[i]) $(f_ref[i]) $(g_ref[i]) $(f_opt[i]) $(σ_opt[i]) $(g_opt[i])")
    end
end

# Fermi function 
begin
    m = models[end]
    p = plot(; title="Fermi function")
    plot!(p; xlim=[μ - 5 / β, μ + 5 / β])
    plot!(
        p, m.x, fermi_fn.(m.x, β, μ); label="Reference", color=:red, legend=:topleft, lw=1
    )
    plot!(p, m.x, model_fermi(m.x, m.θ_sp2); label="SP2 guess", color=:green)
    plot!(
        p, m.x, model_fermi(m.x, m.θ_fermi); label="Model", color=:black, ls=:dashdot, lw=2
    )
end

# Fermi error, Float64
begin
    p = plot(; title="Fermi error, Float64")
    plot!(p; xlim=[μ - 40 / β, μ + 40 / β])
    for m in models
        err = model_fermi(m.x, m.θ_fermi) - fermi_fn.(m.x, β, μ)
        label = @sprintf("Iters = %dk, RMSE = %.1e", m.max_iter / 1000, rmse(m.x, err))
        plot!(p, m.x, err; label)
    end
    p
end

# Fermi error, Float32
begin
    p = plot(; title="Fermi error, Float32")
    plot!(p; xlim=[μ - 40 / β, μ + 40 / β])
    for m in models
        err = model_fermi(Float32.(m.x), Float32.(m.θ_fermi)) - fermi_fn.(m.x, β, μ)
        label = @sprintf("Iters = %dk, RMSE = %.1e", m.max_iter / 1000, rmse(m.x, err))
        plot!(p, m.x, err; label)
    end
    p
end

# Energy function
begin
    m = models[end]

    g_ref = energy_fn.(m.x, β, μ)
    f_opt = model_fermi(m.x, m.θ_fermi)
    σ_opt = model_entropy(m.x, m.θ_entropy)
    g_opt = @. f_opt * (m.x - μ) - β^-1 * σ_opt

    f_sp2 = model_fermi(m.x, m.θ_sp2)
    σ_sp2 = model_entropy(m.x, m.θ_sp2)
    g_sp2 = @. f_sp2 * (m.x - μ) - β^-1 * σ_sp2

    p = plot(; xlim=[μ - 5 / β, μ + 5 / β], ylim=[-5 / β, 1 / β], title="Energy function")
    plot!(p, m.x, g_ref; label="Reference", color=:red, legend=:topleft, lw=1)
    plot!(p, m.x, g_sp2; label="SP2 guess", color=:green)
    plot!(p, m.x, g_opt; label="Model", color=:black, ls=:dashdot, lw=2)
end

# Energy error
begin
    p = plot(; title="Energy error, Float64")
    # plot!(p, xlim=[μ-40/β, μ+40/β])
    for m in models
        g_ref = energy_fn.(m.x, β, μ)
        f_opt = model_fermi(m.x, m.θ_fermi)
        σ_opt = model_entropy(m.x, m.θ_entropy)
        g_opt = @. f_opt * (m.x - μ) - β^-1 * σ_opt
        err = g_opt - g_ref
        label = @sprintf("Iters = %dk, RMSE = %.1e", m.max_iter / 1000, rmse(m.x, err))
        plot!(p, m.x, err; label)
    end
    p
end

# Entropy error
begin
    p = plot(; title="Entropy error, Float64")
    # plot!(p, xlim=[μ-40/β, μ+40/β])
    for m in models
        err = model_entropy(m.x, m.θ_entropy) - entropy_fn.(m.x, β, μ)
        label = @sprintf("Iters = %dk, RMSE = %.1e", m.max_iter / 1000, rmse(m.x, err))
        plot!(p, m.x, err; label)
    end
    p
end

# SP2 roundoff error, Float32/Float64
begin
    p = plot(; title="SP2 roundoff error, Float32/Float64")
    plot!(p; xlim=[μ - 40 / β, μ + 40 / β])

    m = models[end]
    f_fp32 = model_fermi(Float32.(m.x), Float32.(m.θ_sp2))
    f_fp64 = model_fermi(m.x, m.θ_sp2)
    nlayers = length(m.θ_sp2) ÷ layer_width
    plot!(p, m.x, f_fp32 - f_fp64; label="num_layers = $nlayers")
end

### Test sampling of x points

β = 400
μ = 0.5
(θ_sp2, θ_fermi, θ_entropy, x) = generate_model(; β, μ, max_iter=5000, npts_scale=1)

# For testing purposes, finer collection of sample points
x′ = 0:1e-4:1

# Fermi function unoptimized/optimized
rmse(x′, fermi_fn.(x′, β, μ) - model_fermi(x′, θ_sp2))
rmse(x′, fermi_fn.(x′, β, μ) - model_fermi(x′, θ_fermi))
# check sample points
rmse(x, fermi_fn.(x, β, μ) - model_fermi(x, θ_fermi))

# Entropy function unoptimized/optimized
rmse(x′, entropy_fn.(x′, β, μ) - model_entropy(x′, θ_sp2))
rmse(x′, entropy_fn.(x′, β, μ) - model_entropy(x′, θ_entropy))

# Plot Fermi error
begin
    p = plot(; xlim=[μ - 50 / β, μ + 50 / β])
    plot!(p, x′, model_fermi(x′, θ_fermi) - fermi_fn.(x′, β, μ); label="Model error")
    plot!(
        p, x, model_fermi(x, θ_fermi) - fermi_fn.(x, β, μ); label="Discretized model error"
    )
end

# Print model
begin
    θ = reshape(θ_entropy, layer_width, :)
    for i in axes(θ, 2)
        println(θ[:, i])
    end
end

### Test application to matrix

β = 400
μ = 0.5
(θ_sp2, θ_fermi, θ_entropy, x) = generate_model(; β, μ, max_iter=5_000, npts_scale=1)

# Build NxN random matrix
N = 100
H = randn(N, N)
H = (H + H') / 2

# Random matrix eigenvalues empirically satisfy Wigner semicircle distribution.
# It seems safe to say that -span < λ < span, where shift of 0.2 gives some buffer
# for small N
span = (1.4 + 0.2) * sqrt(N)

# Rescaled eigenvalues between 0 and 1
H = (H + span * I) / 2span

f_ref = GenSP2.matrix_fn(x -> fermi_fn(x, β, μ), H)

f_mod64 = GenSP2.fermi_matrix(H, θ_fermi)
maximum(abs.(f_ref - f_mod64))
norm(f_ref - f_mod64)

f_mod32 = GenSP2.fermi_matrix(Float32.(H), Float32.(θ_fermi))
maximum(abs.(f_ref - f_mod32))
norm(f_ref - f_mod32)

f_mod64_2 = copy(H)
temp1 = copy(H)
temp2 = copy(H)
GenSP2.fermi_matrix!(f_mod64_2, temp1, temp2, H, θ_fermi)
norm(f_ref - f_mod64_2)

################

using Plots

x = 0:0.002:1

f1(x) = x^2
f2(x) = 2x - x^2

begin
    p = plot(x, x .^ 2)
    plot!(p, x, @. 2x - x^2)
    plot!(; size=(300, 300), legend=false, dpi=200)
    p
end

begin
    p = plot(x, @. 1 - f2(f1(f2(f1(x)))))
    plot!(; size=(300, 300), legend=false, dpi=200)
    p
end

b = [true, false, true, false]
μ, β = GenSP2.calc_μ_sp2(b)

μ = 0.5680223167828353
# β = 9.423203797761618
β = 20
(θ_sp2, θ_fermi, θ_entropy, x) = generate_model(;
    β, μ, nlayers=4, max_iter=5_000, npts_scale=1
)

fermi(x, μ, β) = 1 / (1 + exp(β * (x - μ)))
free_energy(x, μ, β) = -β^(-1) * log(1 + exp(-β * (x - μ)))

begin
    p = plot(x, @. 1 - f2(f1(f2(f1(x)))))
    plot!(p, x, fermi.(x, μ, β))
    plot!(p, x, model_fermi(x, θ_fermi); color="black", linestyle=:dashdot)
    plot!(; size=(330, 300), legend=false, dpi=200)
    p
end

begin
    f_opt = model_fermi(x, θ_fermi)
    s_opt = model_entropy(x, θ_entropy)
    g_opt = @. f_opt * (x - μ) - β^-1 * s_opt

    p = plot(x, energy_fn.(x, β, μ); color="red")
    plot!(p, x, g_opt; color="black", linestyle=:dashdot)
    plot!(; size=(330, 300), legend=false, dpi=200)
    p
end