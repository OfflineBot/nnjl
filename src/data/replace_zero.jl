
export replace_zero!

replace_zero!(A::Matrix{Float32}, target::Float32) = A .= ifelse.(A .== 0, target, A)

