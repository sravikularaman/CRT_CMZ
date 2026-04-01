"""
Julia tools for cosmic-ray transport equation in cyclindrical geometry and log-spaced grids.
Author: S. Ravikularaman
Last modified: 01-04-2026

Julia note for Python users: Array-indexing starts at 1 (not 0; .= is broadcasting assignment; Func.(...) broadcasts function over arrays (equivalent to np.vectorize)
"""


# Extends radial grid by adding a negative reflection point at the lower boundary
# Inputs: x - vector of radial grid points (r_1, r_2, ..., r_M-1, r_M)
# Outputs: r_i-1 and r_i+1 vectors
function extend_r(x::Vector)

    nx = length(x)

    xl = Vector{Float64}(undef, nx-1) # r_-1, r_1, ..., r_M-3, r_M-2
    xl[1] = - x[1]
    xl[2:end] .= x[1:end-2]

    xu = Vector{Float64}(undef, nx-1) # r_2, r_3, ..., r_M-1, r_M
    xu[1:end] .= x[2:end]

    return xl, xu
end

# Extends axial grid
# Inputs: x - vector of axial grid points (z_1, z_2, ..., z_M-1, z_M)
# Outputs: z_j-1 and z_j+1 vectors
function extend_z(x::Vector)

    nx = length(x)

    xl = Vector{Float64}(undef, nx-2) # z_1, z_2, ..., z_M-3, z_M-2
    xl[1:end] .= x[1:end-2]

    xu = Vector{Float64}(undef, nx-2) # z_3, z_4, ..., z_M-1, z_M
    xu[1:end] .= x[3:end]

    return xl, xu
end

# Extends momentum grid 
# Inputs: x - vector of momentum grid points (p_1, p_2, ..., p_M-1, p_M)
# Outputs: p_k+1 vector
function extend_p(x::Vector)

    nx = length(x)

    xu = Vector{Float64}(undef, nx-1) # p_2, p_3, ..., p_M-1, p_M
    xu[1:end] .= x[2:end]

    return xu
end

# Extends momentum grid with both lower and upper bounds
# Inputs: x - vector of momentum grid points (p_1, p_2, ..., p_M-1, p_M)
# Outputs: p_k-1 and p_k+1 vectors
function extend_m(x::Vector)

    nx = length(x)

    xl = Vector{Float64}(undef, nx-2) # p_1, p_2, ..., p_M-3, p_M-2
    xl[1:end] .= x[1:end-2]

    xu = Vector{Float64}(undef, nx-2) # p_3, p_4, ..., p_M-1, p_M
    xu[1:end] .= x[3:end]

    return xl, xu
end


# Computes differences for radial grid spacing
# Inputs: x - vector of radial grid points (r_1, r_2, ..., r_M-1, r_M)
# Outputs: lower differences, center differences, and upper differences
function delta_r(x::Vector)

    nx = length(x)

    delta_xl = Vector{Float64}(undef, nx-1) # r_1 - r_-1, ..., r_i - r_i-1, ..., r_M-1 - r_M-2
    delta_xl[1] = 2 * x[1] 
    delta_xl[2:end] .= x[2:end-1] .- x[1:end-2]

    delta_xu = Vector{Float64}(undef, nx-1) # r_2 - r_1, ..., r_i+1 - r_i, ..., r_M - r_M-1
    delta_xu[1:end] .= x[2:end] .- x[1:end-1]

    delta_xc = Vector{Float64}(undef, nx-1) # r_2 - r_-1, ..., r_i+1 - r_i-1, ..., r_M - r_M-2
    delta_xc[1] = x[2] + x[1]
    delta_xc[2:end] .= x[3:end] .- x[1:end-2]

    return delta_xl, delta_xc, delta_xu
end


# Computes differences for axial grid spacing
# Inputs: x - vector of axial grid points (z_1, z_2, ..., z_M-1, z_M)
# Outputs: lower differences, center differences, and upper differences
function delta_z(x::Vector)

    nx = length(x)

    delta_xl = Vector{Float64}(undef, nx-2) # z_2 - z_1, ..., z_j - z_j-1, ..., z_M-1 - z_M-2
    delta_xl[1:end] .= x[2:end-1] .- x[1:end-2]

    delta_xu = Vector{Float64}(undef, nx-2) # z_3 - z_2, ..., z_j+1 - z_j, ..., z_M - z_M-1
    delta_xu[1:end] .= x[3:end] .- x[2:end-1]

    delta_xc = Vector{Float64}(undef, nx-2) # z_3 - z_1, ..., z_j+1 - z_j-1, ..., z_M - z_M-2
    delta_xc[1:end] .= x[3:end] .- x[1:end-2]

    return delta_xl, delta_xc, delta_xu 
end


# Computes differences for momentum grid spacing
# Inputs: x - vector of momentum grid points (p_1, p_2, ..., p_M-1, p_M)
# Outputs: center differences only
function delta_p(x::Vector)

    nx = length(x)

    delta_xu = Vector{Float64}(undef, nx-1) # p_2 - p_1, ..., p_k+1 - p_k, ..., p_M - p_M-1
    delta_xu[1:end] .= x[2:end] .- x[1:end-1]

    return delta_xu
end


# Computes differences for momentum grid spacing with both lower and upper bounds
# Inputs: x - vector of momentum grid points (p_1, p_2, ..., p_M-1, p_M)
# Outputs: lower differences, center differences, and upper differences
function delta_m(x::Vector)

    nx = length(x)

    delta_xl = Vector{Float64}(undef, nx-2) # p_2 - p_1, ..., p_k - p_k-1, ..., p_M-1 - p_M-2
    delta_xl[1:end] .= x[2:end-1] .- x[1:end-2]

    delta_xu = Vector{Float64}(undef, nx-2) # p_3 - p_2, ..., p_k+1 - p_k, ..., p_M - p_M-1
    delta_xu[1:end] .= x[3:end] .- x[2:end-1]

    delta_xc = Vector{Float64}(undef, nx-2) # p_3 - p_1, ..., p_k+1 - p_k-1, ..., p_M - p_M-2
    delta_xc[1:end] .= x[3:end] .- x[1:end-2]

    return delta_xl, delta_xc, delta_xu
end