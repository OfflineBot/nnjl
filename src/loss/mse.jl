
export mse 

function mse(y_pred::Matrix{Float32}, y_true::Matrix{Float32})::Float32
    diff = y_pred .- y_true
    return sum(diff .^ 2) / length(diff)
end

