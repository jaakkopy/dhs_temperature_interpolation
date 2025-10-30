using LinearAlgebra
using SparseArrays
using SpecialFunctions

"""Scaling factor for the covariance approximation using the SPDE method"""
function calculate_matern_scaling_tau2(nu::Real, d::Int, kappa::Real, sigma::Real)
    return sqrt(gamma(nu) / (gamma(nu + d/2) * (4*pi)^(d/2))) / (sigma * kappa^(nu))
end

"""Matérn covariance approximation using the SPDE method (Lingren et al. 2011)"""
function create_precision_from_MG(M::AbstractMatrix, G::AbstractMatrix, 
        kappa::Real, nu::Real, d::Real, sigma::Real, alpha_int::Int)
    # M is mass matrix, G is stiffness matrix (∫ ψ' ψ')
    # build K = kappa^2 * M + G
    K = kappa^2 * M + G

    tau2 = calculate_matern_scaling_tau2(nu, d, kappa, sigma)

    if alpha_int == 1
        return Symmetric(tau2 * K)
    elseif alpha_int == 2
        facM = factorize(M)
        return Symmetric(tau2 * (K' * (facM \ K)))
    else
        # recurrence Q_alpha = K * (M \ Q_{alpha-2}) * (M \ K)
        Qm2 = create_precision_from_MG(M, G, kappa, nu, d, sigma, alpha_int - 2)
        facM = factorize(M)
        return Symmetric(K * (facM \ (Qm2 * (facM \ K))))
    end
end