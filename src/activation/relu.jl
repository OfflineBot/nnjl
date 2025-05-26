
export relu

relu = Activation(
    x -> ifelse.(x .<= 0, 0.01f0, x),
    x -> ifelse.(x .<= 0, 0.0f0, 1.0f0)
)

