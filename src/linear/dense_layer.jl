
export DenseLayer

mutable struct DenseLayer
    weight::Matrix{Float32}
    bias::Matrix{Float32}
    z::Union{Matrix{Float32}, Nothing}      # Forwarded Matrix
    a::Union{Matrix{Float32}, Nothing}      # Activated Forwarded Matrix
    activation::Activation

    input::Union{Matrix{Float32}, Nothing}
    grad_weight::Union{Matrix{Float32}, Nothing}
    grad_bias::Union{Matrix{Float32}, Nothing}

    ### Function must take Matrix{Float32} as input and returns Matrix{Float32}
    function DenseLayer(input_size::Int, output_size::Int, activation::Activation)
        weight = randn(input_size, output_size) .* 0.01f0
        bias = randn(1, output_size) .* 0.01f0

        z = nothing
        a = nothing

        input = nothing
        grad_weight = nothing
        grad_bias = nothing

        return new(weight, bias, z, a, activation, input, grad_weight,grad_bias)
    end
end


