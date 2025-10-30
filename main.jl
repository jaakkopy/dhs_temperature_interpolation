using LinearAlgebra
using SparseArrays
using Plots

include("fem_functions.jl")
include("matern_precision.jl")

"""
Test network:

       c
       ^
      /
a --> b
      \
       v
	    d

index <-> edge mapping with shared inlet node convention:

1:na <-> (a,b)
na+1:na+nc <-> (b,c)
na+nc+1:na+nc+nd <-> (b,d)
"""

# General parameters
nedges = 3
edges = [(:a, :b), (:b, :c), (:b, :d)] # symbolic definition for edges
l = [8.0, 10.0, 12.0] # length of edges ab, bc, bd
h = 0.5 # grid spacing
v = [10.0, 8.0, 7.0] # velocities on edges (assume constant)
D = [0.05, 0.06, 0.04] # Diffusion coefficients
alpha = [2.0, 1.5, 1.2] # reaction coefficients
tau = h./(2*v)
dt = 1.0
n = Int.(floor.(l ./ h)) .+ 1
ntotal = sum(n)
uext = 5*ones(sum(n))

# Precision and Matérn parameters
d_matern = 1
alpha_matern = 2
nu_matern = alpha_matern - d_matern/2
ell_matern = 5.0
kappa_matern = sqrt(8*nu_matern)/ell_matern
sigma_matern = 1.0

# Graph level FEM matrix construction
global_idx_counter = 0
vertex_to_global_node = Dict{Symbol, Int}()
edge_local_to_global = Vector{Vector{Int}}(undef, nedges) # mapping from local index i -> global index j for each edge

for i = 1:nedges
    ni = n[i]
    edge_local_to_global[i] = zeros(Int, ni)
    vL, vR = edges[i] # node symbols
    # add nodes if not present:
    if !haskey(vertex_to_global_node, vL)
        global_idx_counter += 1
        vertex_to_global_node[vL] = global_idx_counter
    end
    if !haskey(vertex_to_global_node, vR)
        global_idx_counter += 1
        vertex_to_global_node[vR] = global_idx_counter
    end
    edge_local_to_global[i][1] = vertex_to_global_node[vL] # local inlet node index matches vL index
    edge_local_to_global[i][ni] = vertex_to_global_node[vR] # same for vR and outlet index
    # interior nodes are independent and get own indices:
    for j in 2:(ni-1)
        global_idx_counter += 1
        edge_local_to_global[i][j] = global_idx_counter
    end
end

ntotal_global = global_idx_counter

M_graph = spzeros(ntotal_global, ntotal_global)
G_graph = spzeros(ntotal_global, ntotal_global)

for i in 1:nedges
    local_n = n[i]
    local_M = psi_psi(local_n, h)
    local_G = dpsi_dpsi(local_n, h)
    idxs = edge_local_to_global[i]
    # add local into global
    for a = 1:local_n, b = 1:local_n
        ia = idxs[a]; ib = idxs[b]
        M_graph[ia, ib] += local_M[a,b]
        G_graph[ia, ib] += local_G[a,b]
    end
end

# Graph level operator
L_graph = spzeros(ntotal_global, ntotal_global)
for i in 1:nedges
    local_n = n[i]
    tau = h/(2*v[i])
    local_M_supg = psi_psi(local_n, h) + tau*v[i]*dpsi_psi(local_n)
    local_K_supg = dpsi_dpsi(local_n, h)
    local_A_supg = psi_dpsi(local_n) + tau*v[i]*local_K_supg
    idxs = edge_local_to_global[i]
    # add local into global
    for a = 1:local_n, b = 1:local_n
        ia = idxs[a]; ib = idxs[b]
        L_graph[ia,ib] += v[i]*local_A_supg[a,b] + D[i]*local_K_supg[a,b] + alpha[i]*local_M_supg[a,b]
    end
end

# Free DOF matrices
supply_node_indices = [] # optionally prescribe inlet temperature
f_idx = setdiff(1:ntotal_global, supply_node_indices)
nfree = length(f_idx)
b_idx = setdiff(collect(1:ntotal_global), f_idx)
Mff = M_graph[f_idx, f_idx]
Mfb = M_graph[f_idx, b_idx]
Lff = L_graph[f_idx, f_idx]
Lfb = L_graph[f_idx, b_idx]

Afree = Mff + dt*Lff
FAfree = lu(Afree)
if length(b_idx) > 0
    Dirichlet_temps = [65.0]
    b = -(Mfb + dt*Lfb)*Dirichlet_temps
else
    b = zeros(nfree)
end

function construct_process_noise_covariance_with_ell(ell_matern)
    kappa = sqrt(8*nu_matern)/ell_matern
    Qgraph = create_precision_from_MG(M_graph, G_graph, kappa, nu_matern, d_matern, sigma_matern, alpha_matern)
    Qgraph_free = Qgraph[f_idx, f_idx]
    Qfree = FAfree \ ((dt*Qgraph_free) \ (FAfree' \ I(nfree))) # same as FAfree \ (dt * Cff) * (FAfree' \ I(nfree))
    return Qfree
end

Qfree = construct_process_noise_covariance_with_ell(5.0)

# Measurement operator
m = 2
H = zeros(m, nfree)
# Assume measurements at nodes c and d
c_ind_global = vertex_to_global_node[:c]
c_ind_local = findfirst(==(c_ind_global), f_idx)
d_ind_global = vertex_to_global_node[:d]
d_ind_local = findfirst(==(d_ind_global), f_idx)
H[1, c_ind_local] = 1
H[2, d_ind_local] = 1

# Kalman filter parameters
Niter = 10
u = zeros(nfree)
P = 1.0*I(nfree)
u_true = zeros(nfree)
g = uext[f_idx]
rstd = 0.01 # measurement noise std
R = rstd^2*I(m)
u_hist = zeros(Niter, nfree)
y_hist = zeros(Niter, m)
I_nfree = I(nfree)
I_m = I(m)

meas_c(k) = 8# + 1*sin(dt*k / (10.0*dt*2*pi))
meas_d(k) = 10# + 1*sin(dt*k / (10.0*dt*2*pi))


"""Kalman filtering function. USes global parameters except for process noise covariance"""
function filtering(Qfree)
    u = zeros(nfree)
    P = 1.0*I(nfree)
    rstd = 0.01 # measurement noise std
    m = 2
    R = rstd^2*I(m)
    u_hist = zeros(Niter, nfree)
    I_nfree = I(nfree)
    I_m = I(m)

    meas_c(k) = 8# + 1*sin(dt*k / (10.0*dt*2*pi))
    meas_d(k) = 10# + 1*sin(dt*k / (10.0*dt*2*pi))

    # Filtering loop
    for i = 1:Niter
        # Simulate measurements
        y = [meas_c(i), meas_d(i)] + rstd * randn(m)
        # Predict
        u_pred = FAfree \ Array(Mff*u + dt*g + b)
        P_pred = (FAfree \ P) * (FAfree' \ I_nfree) + Qfree
        # Update
        v = y - H*u_pred
        S = H*P_pred*H' + R
        FS = lu(S)
        K2 = (P_pred * H') * (FS \ I_m)
        u = u_pred + K2*v
        P = (I_nfree - K2 * H) * P_pred * (I_nfree - K2 * H)' + K2 * R * K2' # Joseph form
        # Store
        u_hist[i,:] .= u
    end

    return u_hist, P
end


u_hist, P = filtering(Qfree)

# Plotting
u_std = sqrt.(diag(P))

edge_names = ["(a,b)", "(b,c)", "(b,d)"]

# plot for each edge
p = plot(layout=(nedges,1), size=(500,200*nedges))
for i = 1:nedges
    u_edge = u_hist[end, edge_local_to_global[i]]
    x = range(0, l[i], length=length(u_edge))
    u_edge_std = u_std[edge_local_to_global[i]]
    plot!(p[i], x, u_edge, ribbon=(1.96*u_edge_std, 1.96*u_edge_std), label="mean & 95 % CI", linewidth=2)
    name = edge_names[i]
    title!(p[i], "Edge $name last filtered states")
    xlabel!(p[i], "x [m]")
    ylabel!(p[i], "T [°C]")
end

display(p)
savefig(p, "last_states.pdf")

plots = []

for i = 1:nedges
    u_edge = u_hist[:, edge_local_to_global[i]]
    x = range(0, l[i], length=length(u_edge[1,:]))
    p = heatmap(x, dt*(1:Niter), u_edge)
    name = edge_names[i]
    title!("Edge $name filtered temperature over time")
    xlabel!("x [m]")
    ylabel!("Time [s]")
    push!(plots, p)
end

p = plot(plots..., layout=(nedges,1), size=(500,200*nedges))
display(p)
savefig(p, "temp_fields.pdf")

# Test with different ell
test_ell = [5.0, 20.0, 50.0, 200.0]
nell = length(test_ell)
u_last_varying_ell = []
P_last_varying_ell = []

for i = 1:nell
    Qfree = construct_process_noise_covariance_with_ell(test_ell[i])
    u_hist, P = filtering(Qfree)
    push!(u_last_varying_ell, u_hist[end, :])
    push!(P_last_varying_ell, P)
end

# Plot results with different ell
p = plot(layout=(nedges,1), size=(500,200*nedges))
for j = 1:nell
    u_hist = u_last_varying_ell[j]
    ell = test_ell[j]
    for i = 1:nedges
        u_edge = u_hist[edge_local_to_global[i]]
        x = range(0, l[i], length=length(u_edge))
        plot!(p[i], x, u_edge, label="l=$ell", linewidth=2)
        name = edge_names[i]
        title!(p[i], "Edge $name last filtered states with varying l")
        xlabel!(p[i], "x [m]")
        ylabel!(p[i], "T [°C]")
    end
end

display(p)
savefig(p, "varying_ell.pdf")