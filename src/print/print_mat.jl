
export print_mat
function print_mat(A::AbstractMatrix) 
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
