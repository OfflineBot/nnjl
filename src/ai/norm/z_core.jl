
import Statistics

export z_core

function z_core(input::Matrix{Float32}, no_zero::Float32)::Matrix{Float32}
    μ = Statistics.mean(input, dims=1)
    σ = Statistics.std(input, dims=1)
    replace_zero!(σ, no_zero)
    return (input .- μ) ./ σ
end

