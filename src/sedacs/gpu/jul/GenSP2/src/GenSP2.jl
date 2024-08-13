module GenSP2

include("training.jl")
include("inference.jl")

export read_or_generate_models,
    generate_model,
    energy_fn,
    fermi_fn,
    entropy_fn,
    model_entropy,
    model_fermi,
    sample_weights,
    layer_width

end # module
