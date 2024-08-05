using GenSP2
using LinearAlgebra
using Plots

β = 400
μ = 0.5
(θ_sp2, θ_fermi, θ_entropy, x) = generate_model(; β, μ, max_iter=1_000_000, nlayers=16)

# Print layer weights
display(reshape(θ_fermi, layer_width, :))

maximum(abs.(θ_fermi))

# Higher resolution of sample points for inference
x′ = collect(0:1e-4:1)

begin
    p = plot(xlim=[μ - 50 / β, μ + 50 / β])
    plot!(p, x′, fermi_fn.(x′, β, μ), label="Reference")
    plot!(p, x′, model_fermi(x′, θ_fermi), label="Fitted Model")
    plot!(p, x′, model_fermi(x′, θ_sp2), label="SP2 Model")
end

begin
    p = plot(xlim=[μ - 50 / β, μ + 50 / β])
    plot!(p, x′, model_fermi(x′, θ_fermi) - fermi_fn.(x′, β, μ), label="Model error")
end

begin
    p = plot(xlim=[μ - 10 / β, μ + 10 / β])
    plot!(p, x′, entropy_fn.(x′, β, μ), label="Reference")
    plot!(p, x′, model_entropy(x′, θ_entropy), label="Model")
end

begin
    p = plot(xlim=[μ - 50 / β, μ + 50 / β])
    plot!(p, x′, model_entropy(x′, θ_entropy) - entropy_fn.(x′, β, μ), label="Model error")
end
