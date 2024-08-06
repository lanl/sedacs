using LinearAlgebra
using SparseArrays


function matrix_fn(f, a)
    e = eigen(a)
    e.vectors * diagm(f.(e.values)) * inv(e.vectors)
end

function heaviside_matrix(x, θ)
    npts = length(x)
    θ = reshape(θ, layer_width, :)
    nlayers = size(θ, 2)

    y = x
    Y = zero(x)
    for i = 1:nlayers
        Y += θ[4, i] * y
        y = θ[1, i] * y^2 + θ[2, i] * y + θ[3, i]*I
    end
    Y += y
    return Y
end

function fermi_matrix(x, θ)
    Y = heaviside_matrix(x, θ)
    return I - Y
end


function heaviside_matrix!(res, temp1, temp2, x, θ)
    npts = length(x)
    n = size(x, 2)
    typ = eltype(x)
    θ = reshape(θ, layer_width, :)
    nlayers = size(θ, 2)

    fill!(res, 0.)
    y = temp1
    y² = temp2
    copy!(y, x)

    for i = 1:nlayers
        BLAS.gemm!('N', 'N', one(typ), y, y, zero(typ), y²) # y² = y*y

        BLAS.axpy!(θ[4, i], y, res)
        # @. res += θ[4, i] * y

        BLAS.axpby!(θ[1, i], y², θ[2, i], y) # y = θ₁ y² + θ₂ y
        # @. y = θ[1, i] * y² + θ[2, i] * y

        y[1:n+1:n*n] .+= θ[3, i] # y += θ₃ I
    end

    BLAS.axpy!(1.0, y, res)
    # @. res += y

    return res
end

function fermi_matrix!(res, temp1, temp2, x, θ)
    heaviside_matrix!(res, temp1, temp2, x, θ)

    n = size(x, 2)
    BLAS.scal!(n*n, -1., res, 1) # res *= -1
    res[1:n+1:n*n] .+= 1.0       # res += I

    return res
end
