
export sigmoid

sigmoid = Activation(
    x -> 1.0f0 ./ (1.0f0 .+ exp.(-x)),
    x -> begin
        s = 1.0f0 ./ (1.0f0 .+ exp.(-x))
        s .* (1 .- s)
    end
)

