
export update!

function update!(layer::ConvLayer, learning_rate::Float32)
    layer.kernel .-= layer.gradients * learning_rate
    layer.bias .-= layer.bias_grad .* learning_rate
end

