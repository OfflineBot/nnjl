
module nnjl

include("./ai/ai.jl")
include("./utils/utils.jl")


function run_nn(epoch::Int, iterations::Int, lr::Float32)

    input = [
        1.f0 1.f0;
        0.f0 0.f0;
        0.f0 1.f0;
        1.f0 0.f0
    ]
    output_data_raw = [ 0.f0, 0.f0, 1.f0, 1.f0 ]
    output = reshape(output_data_raw, 4, 1)

    #input = z_core(input_data_raw, 0.01f0)
    layer1 = DenseLayer(2, 2, sigmoid)
    layer2 = DenseLayer(2, 1, sigmoid)

    out_f2 = Matrix{Float32}(undef, (0, 0))

    for i in 1:iterations

        f1 = forward!(layer1, input)
        f2 = forward!(layer2, f1)
        out_f2 = f2
        loss = mse(f2, output)
        if i % (iterations/100) == 0
            println("Epoch: $epoch Iteration: $i Loss: $loss")
        end
        if loss < 0.001f0
            println("Loss Break At $i")
            return i
        end

        d2 = f2 .- output
        d1 = backward!(layer2, d2, nothing)
        backward!(layer1, d1, layer2.weight)

        update!(layer1, lr)
        update!(layer2, lr)

    end
    println(out_f2)
    return -1
end

function epochs(epoch::Int, iterations::Int, lr::Float32)
    losses = 0
    wins = 0
    for i in 1:epoch
        println("$i")
        x = run_nn(i, iterations, lr)
        if x < 0
            losses += 1
        else
            wins += 1
        end
    end

    println("Wins: $wins/$epoch: ",  wins / epoch * 100.0f0, "%")
    println("Losses: $losses/$epoch: ", losses / epoch * 100.0f0, "%")
end

epochs(10, 10_000, 0.01f0)


end

