
export pretty_print

function pretty_print(A::AbstractMatrix) 
    println("[")
    for i in axes(A, 1)
        print("  [ ")
        for j in axes(A, 2)
            print(A[i, j], " ")
        end
        println("]")
    end
    println("]")
end

function pretty_print(A::AbstractArray{<:Any, 3})
    println("[")
    for i in axes(A, 1)
        println("  [ ")
        for j in axes(A, 2)
            print("    [")
            for k in axes(A, 3)
                print(A[i, j, k], " ")
            end
            println("]")
        end
        println("  ]")
    end
    println("]")
end

