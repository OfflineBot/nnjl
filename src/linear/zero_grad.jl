
export zero_grad!

function zero_grad!(layer::DenseLayer)
    layer.grad_weight = nothing
    layer.grad_bias = nothing
end


