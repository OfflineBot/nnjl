
export forward!


function pad_array(input::Array{Float32, 4}, padding::Int)::Array{Float32, 4}
    items, input_size, input_height, input_width = size(input)
    padded = zeros(Float32, items, input_size, input_height + 2 * padding, input_width + 2 * padding)
    padded[:, :, padding+1:padding+input_height, padding+1:padding+input_width] .= input
    return padded

end


function forward!(layer::ConvLayer, input::Array{Float32, 4})::Array{Float32, 4}

    padding = layer.padding
    stride = layer.stride
    items, in_chn, _, _ = size(input)

    if in_chn != layer.in_channel
        error("Input Channels dont match!")
    end

    padded = pad_array(input, padding)
    out_x = floor(Int, (size(padded, 3) - layer.kernel_x) / stride) + 1
    out_y = floor(Int, (size(padded, 4) - layer.kernel_y) / stride) + 1
    out_array = zeros(items, layer.out_channel, out_x, out_y)


    for item in 1:items 
        for out_c in 1:layer.out_channel
            for in_c in 1:in_chn
                for imgx in 1:out_x
                    for imgy in 1:out_y
                        for kx in 1:layer.kernel_x
                            for ky in 1:layer.kernel_y
                                ix = (imgx - 1) * stride + kx
                                iy = (imgy - 1) * stride + ky
                                out_array[item, out_c, imgx, imgy] += 
                                    padded[item, in_c, ix, iy] * layer.kernel[out_c, in_c, kx, ky]
                            end
                        end
                    end
                end
            end
        end
    end

    return out_array
end

