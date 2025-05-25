

export forward!

function forward!(layer::DenseLayer, input::Matrix{Float32})::Matrix{Float32}
    z = input * layer.weight .+ layer.bias
    a = layer.activation.f(z)
    layer.z = z
    layer.a = a
    return a
end

