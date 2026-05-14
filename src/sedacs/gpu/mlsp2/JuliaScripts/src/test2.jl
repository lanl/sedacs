using GenSP2
using LinearAlgebra
using Plots

# Parameters for the model
β = 1500
μ = 1/3
nlayers = 26
@time (θ_sp2, θ_fermi, θ_entropy, x) = generate_model(; β, μ, max_iter=200000, nlayers)

# Higher resolution of sample points for inference
x′ = collect(0:1e-4:1)

rmse = norm(model_fermi(x′, θ_fermi) - fermi_fn.(x′, β, μ)) / sqrt(length(x′))

# SP2 layer weights, Human interpretable form
display(reshape(θ_sp2, layer_width, :))

# Same data in compressed column-major form
println(θ_sp2)

# Fitted parameters for Fermi function
println(θ_fermi)

# Fitted parameters for entropy function
println(θ_entropy)


maximum(abs.(θ_fermi))

begin
    p = plot(xlim=[μ - 50 / β, μ + 50 / β])
    plot!(p, x′, model_fermi(x′, θ_fermi) - fermi_fn.(x′, β, μ), label="Model error")
end

begin
    p = plot(xlim=[μ - 50 / β, μ + 50 / β])
    plot!(p, x′, model_fermi(x′, θ_sp2), label="SP2 Model")
    plot!(p, x′, model_fermi(x′, θ_fermi), label="Fitted Model", color=:pink, linewidth=4)
    plot!(p, x′, fermi_fn.(x′, β, μ), label="Reference", style=:dash, color=:black, linewidth=2)
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


maximum(model_entropy(x′, θ_entropy) - entropy_fn.(x′, β, μ))