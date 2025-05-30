
module nnjl

include("./ai/ai.jl")
include("./utils/utils.jl")

input::Array{Float32, 4} = rand(Float32, 2, 2, 9, 9)

c = ConvLayer(2, 2, 3, 3, 0, 1)

x = forward!(c, input)

println(size(input))
println(size(x))

end

