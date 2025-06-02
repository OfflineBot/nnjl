
export backward!

function backward!(layer::ConvLayer, last_delta::Array{Float32, 4})::Array{Float32, 4}
    layer.gradients = cross_correlation(layer.input, last_delta)
    layer.bias_grad = sum(last_delta, dims=(1, 3, 4))[:]
    new_delta = conv_transpose(last_delta, layer.kernel)
    return new_delta
end


function cross_correlation(input::Array{Float32, 4}, last_delta::Array{Float32, 4})::Array{Float32, 4}

end


