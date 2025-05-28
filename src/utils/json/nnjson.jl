
export NNJsonObject

mutable struct NNJsonStruct
    input_size::Int
    output_size::Int
    activation_name::String
    weights::Matrix{Float32}
    bias::Matrix{Float32}

    function NNJsonObject(input_size::Int, output_size::Int, activation_name::String, weights::Matrix{Float32}, bias::Matrix{Float32})
        new(input_size, output_size, activation_name, weights, bias)
    end
end


