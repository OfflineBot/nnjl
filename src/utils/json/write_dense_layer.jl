
using JSON

export write_json_dense_layer

### name must end on .json (e.g. "layer1.json", "x.json")
function write_json_dense_layer(layer::DenseLayer, name::String)
    weight_sizes = size(layer.weight)
    data = NNJson(weight_sizes[1], weight_sizes[2], layer.activation.name, layer.weight, layer.bias)
    if !isdir("data")
        mkdir("data")
    end
    open(string("data/", name), "w") do io
        json_str = JSON.json(data, 4)
        write(io, json_str)
    end
end

