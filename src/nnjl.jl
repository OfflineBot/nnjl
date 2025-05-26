
module nnjl

include("./activation/activation.jl")
include("./data/data.jl")
include("./linear/linear.jl")
include("./print/print_mat.jl")
include("./loss/loss.jl")



function iterate(num::Int, epoch::Int)

    input::Matrix{Float32} = Float32[
        1 1;
        0 0;
        1 0;
        0 1
    ]

    output_vec = Float32[0, 0, 1, 1]
    output = reshape(output_vec, 4, 1)

    replace_zero!(input, 0.001f0)
    replace_zero!(output, 0.001f0)


    layer1 = DenseLayer(2, 10, relu)
    layer2 = DenseLayer(10, 1, relu)
    loss_break = 0.001f0
    lr = 0.01f0

    f2 = Matrix{Float32}(undef, (0, 0))
    for i in 0:num
        f1 = forward!(layer1, input)
        f2 = forward!(layer2, f1)

        loss = mse(f2, output)
        if i % 10_000 == 0
            println("Epoch: [$epoch] Iteration: [$i/$num] Loss: [$loss]")
        end
        if loss < loss_break
            println("loss break at i: $i")
            return true, f2
        end

        d2 = f2 .- output
        d1 = backward!(layer2, d2, nothing)
        backward!(layer1, d1, layer2.weight)

        update!(layer1, lr)
        update!(layer2, lr)

        # Optional
        zero_grad!(layer1)
        zero_grad!(layer2)
    end

    return false, f2
end

function test()
    finishes = 0
    output_vec = Float32[0, 0, 1, 1]
    output = reshape(output_vec, 4, 1)

    losses = 0
    out_matrix = output

    for i in 0:20
        out, f = iterate(100_000, i)
        out_matrix .+= f
        if out === true
            finishes += 1
        else
            losses += 1
        end
    end

    out_matrix ./= 20.0f0

    println("Iterations: 20")
    println("Finishes: ", finishes)
    println("Losses: ", losses)
    print_pretty_matrix(out_matrix)
end

test()
end

