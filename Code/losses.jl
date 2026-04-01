"""
Julia function for energy and momentum loss rates for protons and electrons, including ionisation, bremsstrahlung, synchrotron, and inverse Compton losses. Also includes functions to compute loss timescales.
Author: S. Ravikularaman
Last modified: 01-04-2026

Proton loss rates: ionisation + p-p collisions from Fig. 1 Padovani 2018 (https://www.aanda.org/articles/aa/full_html/2018/06/aa32202-17/aa32202-17.html) in untis log10(E/eV) and log10(loss rate/eV cm2)
Electron loss rates: ionisation and bremsstrahlung from Padovani 2018 in above units, synchrotron loss rates computed using standard formula, inverse Compton loss rates computed using Khangulyan et al. 2014 (https://iopscience.iop.org/article/10.1088/0004-637X/783/2/100)
"""

using DelimitedFiles
using Interpolations
using LinearAlgebra


const m_p = 938.272 # in MeV, proton mass
const m_e = 0.511 # in MeV, electron mass
const c = 3e10 # cm/s, velocity of light
const sigma_T = 6.653e-25 #cm2, Thomson scattering cross section
const m_el = 9.1094e-28 #g
const e = 4.8032e-10 # in cm^3/2 g^1/2 s-1
const r_0 = e^2/(m_el*(c^2)) # cm electron radius
const h_bar_cgs = 1.0546e-27 #cm2 g s-1
const k_B = 8.6173e-11 # MeV K-1


# Computes velocity, relativistic beta, and Lorentz factor gamma 
# Inputs: E_kin = kinetic energy in MeV, rest_mass = rest mass in MeV
# Outputs: velocity in cm/s, beta (dimensionless), and gamma (dimensionless)
function velocity(E_kin::Float64, rest_mass::Float64)

    gamma = 1 + (E_kin / rest_mass)
    beta = sqrt(1 - (1 / (gamma^2) ) )
    v = c * beta

    return v, beta, gamma
end


#-------------------------------------PROTONS-------------------------------------#

lossp = readdlm(joinpath(@__DIR__, "lossfctp.txt"), ',')
lp = linear_interpolation(lossp[1:end-4, 1], lossp[1:end-4, 2], extrapolation_bc=Line())


# Computes proton energy loss rate due to ionization
# Inputs: E = kinetic energy in MeV, n = ambient density in cm^-3
# Outputs: energy loss rate in MeV/s
function E_p_dot(E::Float64, n::Float64)

    v, _, _ = velocity(E, m_p)
    loss_rate = (10^( lp(log10(E * 1e6)) )) * 1e-6
    
    return (1/2) * v * n * loss_rate
end


# Computes proton momentum loss rate due to ionization
# Inputs: p = momentum in MeV/c, n = ambient density in cm^-3
# Outputs: momentum loss rate in MeV/c/s
function p_p_dot(p::Float64, n::Float64)

    total_E = sqrt(p^2 + m_p^2)
    E = total_E - m_p
    dp_on_dT = total_E / p
    E_dot = E_p_dot(E, n)
    p_dot = dp_on_dT * E_dot

    return p_dot
end


# Computes proton energy loss timescale (time for particle to lose all kinetic energy)
# Inputs: E = kinetic energy in MeV, n = ambient density in cm^-3
# Outputs: energy loss timescale in s
function time_loss_p(E::Float64, n::Float64)
    return E/E_p_dot(E, n)
end


# Computes proton momentum loss timescale
# Inputs: p = momentum in MeV/c, n = ambient density in cm^-3
# Outputs: momentum loss timescale in s
function tau_loss_p(p::Float64, n::Float64)
    total_E = sqrt(p^2 + m_p^2)
    E = total_E - m_p
    tau = time_loss_p(E, n)
    return tau
end


#-------------------------------------ELECTRONS-------------------------------------#

le_ion_data = readdlm(joinpath(@__DIR__, "losse_ion.txt"), ',')
le_ion = linear_interpolation(le_ion_data[1:end, 1], le_ion_data[1:end, 2],  extrapolation_bc=Line())

le_bs_data = readdlm(joinpath(@__DIR__, "losse_bs.txt"), ',')
le_bs = linear_interpolation(le_bs_data[1:end, 1], le_bs_data[1:end, 2],  extrapolation_bc=Line())


# Computes electron energy loss rate due to ionization
# Inputs: E = kinetic energy in MeV, n = ambient density in cm^-3
# Outputs: energy loss rate in MeV/s
function E_e_dot_ion(E::Float64, n::Float64)
    v, _, _ = velocity(E, m_e)
    loss_rate = (10^(le_ion(log10(E * 1e6)))) * 1e-6
    return (1/2) * v * n * loss_rate
end


# Computes electron momentum loss rate due to ionization
# Inputs: p = momentum in MeV/c, n = ambient density in cm^-3
# Outputs: momentum loss rate in MeV/c/s
function p_e_dot_ion(p::Float64, n::Float64)
    total_E = sqrt(p^2 + m_e^2)
    E = total_E - m_e
    dp_on_dT = total_E/p
    E_dot = E_e_dot_ion(E, n)
    p_dot = dp_on_dT*E_dot
    return p_dot
end


# Computes electron energy loss rate due to bremsstrahlung
# Inputs: E = kinetic energy in MeV, n = ambient density in cm^-3
# Outputs: energy loss rate in MeV/s
function E_e_dot_bs(E::Float64, n::Float64)
    v, _, _ = velocity(E, m_e)
    loss_rate = (10^(le_bs(log10(E * 1e6)))) * 1e-6
    return (1/2) * v * n * loss_rate
end


# Computes electron momentum loss rate due to bremsstrahlung
# Inputs: p = momentum in MeV/c, n = ambient density in cm^-3
# Outputs: momentum loss rate in MeV/c/s
function p_e_dot_bs(p::Float64, n::Float64)
    total_E = sqrt(p^2 + m_e^2)
    E = total_E - m_e
    dp_on_dT = total_E/p
    E_dot = E_e_dot_bs(E, n)
    p_dot = dp_on_dT*E_dot
    return p_dot
end


# Computes electron energy loss rate due to synchrotron radiation
# Inputs: E = kinetic energy in MeV, B_uG = magnetic field in microGauss
# Outputs: energy loss rate in MeV/s
function E_e_dot_syn(E::Float64, B_uG::Float64)
    B_G = 1e-6 * B_uG
    B = 1e-4 * B_G
    mu_0 = 4 * pi * 1e-7 # Tm/A 
    B_2_2_mu = B^2 / (2 * mu_0) * 6.242e6 #TA/m = J -> MeV/cm3
    _, beta, gamma = velocity(E, m_e)
    loss_rate = (4/3) * sigma_T * c * (B_2_2_mu) * (beta^2) * (gamma^2)
    return loss_rate
end


# Computes electron momentum loss rate due to synchrotron radiation
# Inputs: p = momentum in MeV/c, B_uG = magnetic field in microGauss
# Outputs: momentum loss rate in MeV/c/s
function p_e_dot_syn(p::Float64, B_uG::Float64)
    total_E = sqrt(p^2 + m_e^2)
    E = total_E - m_e
    dp_on_dT = total_E/p
    E_dot = E_e_dot_syn(E, B_uG)
    p_dot = dp_on_dT*E_dot
    return p_dot
end


# Helper function for inverse Compton scattering (dimensionless energy ratio dependent)
# Inputs: u = dimensionless energy ratio
# Outputs: g factor (dimensionless)
function g_iso(u::Float64)
    a_i = -0.362
    b_i = 0.826
    alpha_i = 0.682
    beta_i = 1.281
    g = 1.0 + ((a_i * (u^alpha_i)) / (1.0 + (b_i * (u^beta_i))))
    return 1 / g
end


# Helper function for inverse Compton scattering (isotropic photon field)
# Inputs: u = dimensionless energy ratio
# Outputs: G_0 factor (dimensionless)
function G_iso_0(u::Float64)
    c_iso = 5.68
    num = c_iso * u * log(1.0 + (0.722 * u / c_iso))
    den = 1.0 + (c_iso * u / 0.822)
    return num / den
end


# Helper function for inverse Compton scattering (combines isotropic components)
# Inputs: u = dimensionless energy ratio
# Outputs: G factor (dimensionless)
function G_iso(u::Float64)
    return G_iso_0(u) * g_iso(u)
end


# Computes electron energy loss rate due to inverse Compton scattering with isotropic photon field
# Inputs: E = kinetic energy (MeV), T = photon temperature (K), k_dil = dilution factor
# Outputs: energy loss rate in MeV/s
function E_e_dot_IC(E::Float64, T::Float64, k_dil::Float64)
    E = E / m_e
    T = k_B * T / m_e
    t = 4 * E * T
    pre_num = 2 * r_0^2 * m_el^3 * c^4 * k_dil * T^2
    pre_den = pi * h_bar_cgs^3 
    pre = pre_num / pre_den
    return pre * G_iso(t) * m_e 
end


# Computes electron momentum loss rate due to inverse Compton scattering
# Inputs: p = momentum in MeV/c, T = photon temperature (K), k_dil = dilution factor
# Outputs: momentum loss rate in MeV/c/s
function p_e_dot_IC(p::Float64, T::Float64, k_dil::Float64)
    total_E = sqrt(p^2 + m_e^2)
    E = total_E - m_e
    dp_on_dT = total_E/p
    E_dot = E_e_dot_IC(E, T, k_dil)
    p_dot = dp_on_dT*E_dot
    return p_dot
end