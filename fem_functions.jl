using SparseArrays

"""(ψ_i, ψ_j)"""
function psi_psi(n::Int, h::Real)
    A = spzeros(n,n)
    for i = 1:n
        A[i,i] = 2*h/3
        if i + 1 <= n
            A[i,i+1] = h/6
            A[i+1,i] = h/6
        end
    end
    A[1,1] = h/3
    A[n,n] = h/3
    return A
end

"""(ψ_i, 1)"""
function psi_1(n::Int, h::Real)
    A = spzeros(n,n)
    for i = 1:n
        A[i,i] = h
    end
    A[1,1] = h/2
    A[n,n] = h/2
    return A
end

"""(ψ'_i, ψ'_j)"""
function dpsi_dpsi(n::Int, h::Real)
    A = spzeros(n,n)
    for i = 1:n
        A[i,i] = 2/h
        if i + 1 <= n
            A[i,i+1] = -1/h
            A[i+1,i] = -1/h
        end
    end
    A[1,1] = 1/h
    A[n,n] = 1/h
    return A
end

"""(ψ_i, ψ'_j)"""
function psi_dpsi(n::Int)
    A = spzeros(n,n)
    for i = 1:n
        if i + 1 <= n
            A[i,i+1] = 1/2
            A[i+1,i] = -1/2
        end
    end
    A[1,1] = -1/2
    A[n,n] = 1/2
    return A
end

"""(ψ'_i, ψ_j)"""
function dpsi_psi(n::Int)
    A = spzeros(n,n)
    for i = 1:n
        if i + 1 <= n
            A[i,i+1] = -1/2
            A[i+1,i] = 1/2
        end
    end
    A[1,1] = -1/2
    A[n,n] = 1/2
    return A
end