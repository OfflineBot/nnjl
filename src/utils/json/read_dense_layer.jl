
import JSON

export read_json_dense_layer

function read_json_dense_layer(name::String, activation::Union{Activation, Nothing})::DenseLayer
    if !isdir("data")
        mkdir("data")
    end

    layer_data = JSON.parsefile(string("data/", name))

    weights = Float32.(hcat(layer_data["weights"]...))
    bias = Float32.(hcat(layer_data["bias"]...))
    activation_function = layer_data["activation_name"]

    if activation !== nothing
        return DenseLayer(weights, bias, activation)
    end

    if activation_function === "relu"
        activation_function = relu
    elseif activation_function === "leaky_relu"
        activation_function = leaky_relu
    elseif activation_function === "sigmoid"
        activation_function = sigmoid
    elseif activation_function === "identity"
        activation_function = identity
    else 
        error("No known activation function. Add your custom one if needed!")
    end

    return DenseLayer(weights, bias, activation_function)
end

