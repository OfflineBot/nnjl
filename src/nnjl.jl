
module nnjl

include("./activation/activation.jl")
include("./linear/linear.jl")
include("./data/data.jl")
include("./print/print_mat.jl")
include("./loss/loss.jl")

in::Int64 = 6
out::Int64 = 8

x = DenseLayer(in, out, relu)

write_json_dense_layer(x, "layer1.json")

y = read_json_dense_layer("layer1.json", nothing)

write_json_dense_layer(y, "layer2.json")

print_pretty_matrix(x.bias)

end

