

export sigmoid
sigmoid(A::Matrix{Float32})::Matrix{Float32} = 1.0f0 ./ (1.0f0 .+ exp.(-A))


export deriv_sigmoid
deriv_sigmoid(A::Matrix{Float32})::Matrix{Float32} = sigmoid(A) .* (1.0f0 .- sigmoid(A))

