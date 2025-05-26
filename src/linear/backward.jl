
export backward!

function backward!(layer::DenseLayer, delta::Matrix{Float32}, weight::Union{Matrix{Float32}, Nothing})::Matrix{Float32}
    if weight === nothing
        delta = delta
    else 
        delta = delta * transpose(weight) .* layer.activation.f_prime(layer.z)
    end
    layer.grad_weight = transpose(layer.input) * delta
    layer.grad_bias = sum(delta, dims=1)
    return delta
end

