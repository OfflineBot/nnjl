
export backward!

function backward!(layer::ConvLayer, last_delta::Array{Float32, 4})::Array{Float32, 4}
    layer.gradients = cross_correlation(layer.input, last_delta, layer.stride, layer.padding, layer)
    layer.bias_grad = sum(last_delta, dims=(1, 3, 4))[:]
    new_delta = conv_transpose(last_delta, layer.kernel, layer.stride, layer.padding)
    return new_delta
end


function cross_correlation(input::Array{Float32, 4}, delta::Array{Float32, 4}, stride::Int, padding::Int, layer::ConvLayer)::Array{Float32, 4}
    items, in_c, in_h, in_w = size(input)
    _, out_c, out_h, out_w = size(delta)

    kH = size(layer.kernel, 3)
    kW = size(layer.kernel, 4)

    grad_kernel = zeros(Float32, out_c, in_c, kH, kW)

    padded_input = pad_array(input, padding)

    for item in 1:items
        for oc in 1:out_c
            for ic in 1:in_c
                for x in 1:out_h
                    for y in 1:out_w
                        for kx in 1:kH
                            for ky in 1:kW
                                ix = (x - 1) * stride + kx
                                iy = (y - 1) * stride + ky
                                grad_kernel[oc, ic, kx, ky] += 
                                    padded_input[item, ic, ix, iy] * delta[item, oc, x, y]
                            end
                        end
                    end
                end
            end
        end
    end

    return grad_kernel
end
