

export relu
relu(A::Matrix{Float32})::Matrix{Float32} = ifelse.(A .<= 0, 0.0f0, A)


export deriv_relu
deriv_relu(A::Matrix{Float32})::Matrix{Float32} = ifelse.(A .<= 0, 0.0f0, 1.0f0)


