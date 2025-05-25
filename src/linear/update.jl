
export update!

function update!(layer::DenseLayer, learning_rate::Float32)
    layer.weight .-= layer.grad_weigth .* learning_rate
    layer.bias .-= layer.grad_bias .* learning_rate
end

