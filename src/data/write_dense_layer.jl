
using JSON

export write_json_dense_layer

### name must end on .json (e.g. "layer1.json", "x.json")
function write_json_dense_layer(layer::DenseLayer, name::String)
    data = Dict(
        "input_size" => size(layer.weight)[1],
        "output_size" => size(layer.weight)[2],
        "weights" => layer.weight,
        "bias" => layer.bias
    )
    if !isdir("data")
        mkdir("data")
    end
    open(string("data/", name), "w") do io
        JSON.print(io, data)
    end
end

