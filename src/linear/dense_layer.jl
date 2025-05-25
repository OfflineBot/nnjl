
export DenseLayer

mutable struct DenseLayer
    weight::Matrix{Float32}
    bias::Vector{Float32}
    z::Matrix{Float32}      # Forwarded Matrix
    a::Matrix{Float32}      # Activated Forwarded Matrix
    activation::Activation

    grad_weigth::Union{Matrix{Float32}, Nothing}
    grad_bias::Union{Matrix{Float32}, Nothing}

    ### Function must take Matrix{Float32} as input and returns Matrix{Float32}
    function DenseLayer(input_size::Int, output_size::Int, activation::Activation)
        weigth = randn(input_size, output_size) .* 0.01f0
        bias = randn(output_size) .* 0.01f0

        z = Matrix{Float32}(undef, (0, 0))
        a = Matrix{Float32}(undef, (0, 0))

        grad_weigth = nothing
        grad_bias = nothing

        return new(weigth, bias, z, a, grad_weigth, grad_bias, activation)
    end
end


