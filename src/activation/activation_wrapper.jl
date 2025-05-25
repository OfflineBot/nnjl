
export Activation

### Both Functions take a Matrix{Float32} as input and return a Matrix{Float32}
struct Activation
    f::Function
    f_prime::Function
end

