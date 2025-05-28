
export pretty_print_matrix

function pretty_print_matrix(A::AbstractMatrix) 
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
