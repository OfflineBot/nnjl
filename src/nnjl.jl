
module nnjl

include("./activation/activation.jl")
include("./data/data.jl")
include("./linear/linear.jl")
include("./print/print_mat.jl")
include("./loss/loss.jl")

x = DenseLayer(3, 2, relu)

write_json_dense_layer(x, "layer1.json")

end

