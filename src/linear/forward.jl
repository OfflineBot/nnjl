
export forward

### Activation must take a Matrix{Float32} as input and return a Matrix{Float32}
forward(
    x::Matrix{Float32}, 
    weight::Matrix{Float32}, 
    bias::Vector{Float32},
    activation::Function = identity
)::Tuple{Matrix{Float32}, Matrix{Float32}} =
    let z = x * weight .+ bias
        (z, activation(z))
    end


