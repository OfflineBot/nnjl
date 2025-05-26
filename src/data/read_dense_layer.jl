
using JSON

export read_json_dense_layer

function read_json_dense_layer(name::String)
    if !isdir("data")
        mkdir("data")
    end
    layer_data = JSON.parsefile(string("data/", name))
    weights = layer_data["weights"]
    weights = layer_data["bias"]
end

