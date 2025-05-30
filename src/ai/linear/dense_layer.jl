
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

    function DenseLayer(weight::Matrix{Float32}, bias::Matrix{Float32}, activation::Activation)
        return new(weight, bias, nothing, nothing, activation, nothing, nothing, nothing)
    end
end

function DenseLayer(input_size::Int, output_size::Int, activation::Activation)
    weight::Matrix{Float32} = randn(input_size, output_size) .* 0.01f0
    bias::Matrix{Float32} = randn(1, output_size) .* 0.01f0
    return DenseLayer(weight, bias, activation)
end


