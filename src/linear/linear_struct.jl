
struct DenseLayer
    weight::Matrix{Float32}
    bias::Vector{Float32}
    z::Matrix{Float32}
    a::Matrix{Float32}

    grad_weigth::Union{Matrix{Float32}, Nothing}
    grad_bias::Union{Matrix{Float32}, Nothing}

    function DenseLayer(input_size::Int, output_size::Int)
        weigth = randn(input_size, output_size) .* 0.01f0
        bias = randn(output_size) .* 0.01f0

        z = Matrix{Float32}(undef, (0, 0))
        a = Matrix{Float32}(undef, (0, 0))

        grad_weigth = nothing
        grad_bias = nothing

        return new(weigth, bias, z, a, grad_weigth, grad_bias)
    end
end


