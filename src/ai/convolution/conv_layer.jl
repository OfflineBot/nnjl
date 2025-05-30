
mutable struct ConvLayer
    out_channel::Int
    in_channel::Int

    kernel_x::Int
    kernel_y::Int
    kernel::Array{Float32, 4}

    padding::Int
    stride::Int

    input::Union{Array{Float32, 4}, Nothing}

    function ConvLayer(out_channel::Int, in_channel::Int, kernel_x::Int, kernel_y::Int, padding::Int, stride::Int)
        kernel::Array{Float32, 4} = randn(out_channel, in_channel, kernel_x, kernel_y) .* 0.01f0
        return new(out_channel, in_channel, kernel_x, kernel_y, kernel, padding, stride, nothing)
    end
end

