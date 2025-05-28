import pybamm
import numpy as np

def graphite_diffusivity_Jeong(sto, T):

    D_ref = 6.9572 * 10 ** (-15)
    E_D_s = 49968
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def graphite_electrolyte_exchange_current_density_Jeong(c_e, c_s_surf, c_s_max, T):
    """
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 3.0866 * 10 ** (-7)  # unit has been converted
    # units are (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 23859
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

def graphite_entropic_change_PeymanMPM(sto, c_s_max):
    """
    Graphite entropic change in open-circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry taken from [1]

    References
    ----------
    .. [1] K.E. Thomas, J. Newman, "Heats of mixing and entropy in porous insertion
           electrode", J. of Power Sources 119 (2003) 844-849

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    du_dT = 10 ** (-3) * (
        0.28
        - 1.56 * sto
        - 8.92 * sto ** (2)
        + 57.21 * sto ** (3)
        - 110.7 * sto ** (4)
        + 90.71 * sto ** (5)
        - 27.14 * sto ** (6)
    )

    return du_dT

def NMC_diffusivity_Jeong(sto, T):

    D_ref = 9.6506e-15
    E_D_s = 11331
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius

def graphite_volume_change_mohtat(sto,c_s_max):
    stoichpoints = np.array([0,0.12,0.18,0.24,0.50,1])
    thicknesspoints = np.array([0,2.406/100,3.3568/100,4.3668/100,5.583/100,13.0635/100])
    x = [sto]
    t_change = pybamm.Interpolant(stoichpoints, thicknesspoints, x, name=None, interpolator='linear', extrapolate=True, entries_string=None)
    return t_change

def graphite_volume_change_mohtat_derivative(sto, c_s_max):
    stoichpoints = np.array([0, 0.12, 0.18, 0.24, 0.50, 1])
    thicknesspoints = np.array([0, 2.406 / 100, 3.3568 / 100, 4.3668 / 100, 5.583 / 100, 13.0635 / 100])
    xnew = np.linspace(0, 1, 100)
    thickness_new = np.interp(xnew, stoichpoints, thicknesspoints)
    gradient_new = []
    for i in range(len(xnew) - 1):
        gradient_new.append((thickness_new[i + 1] - thickness_new[i]) / (xnew[i + 1] - xnew[i]))
    gradient_new.append(gradient_new[-1])
    gradient_new = np.array(gradient_new)

    x = [sto]
    dt_change_dsto = pybamm.Interpolant(xnew,gradient_new,x,name=None, interpolator='linear',extrapolate=True,entries_string=None)
    return dt_change_dsto

def nmc_volume_change_mohtat(sto,c_s_max):
    t_change = -1.10/100*(1-sto)
    return t_change

def NMC_electrolyte_exchange_current_density_Jeong(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 4.6996 * 10 ** (-6)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 49664
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

def NMC_entropic_change_PeymanMPM(sto, c_s_max):
    """
    Nickel Manganese Cobalt (NMC) entropic change in open-circuit potential (OCP) at
    a temperature of 298.15K as a function of the OCP. The fit is taken from [1].

    References
    ----------
    .. [1] W. Le, I. Belharouak, D. Vissers, K. Amine, "In situ thermal study of
    li1+ x [ni1/ 3co1/ 3mn1/ 3] 1- x o2 using isothermal micro-clorimetric
    techniques",
    J. of the Electrochemical Society 153 (11) (2006) A2147–A2151.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    # Since the equation uses the OCP at each stoichiometry as input,
    # we need OCP function here

    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * sto**2
        - 2.0843 * sto**3
        + 3.5146 * sto**4
        - 2.2166 * sto**5
        - 0.5623 * 10 ** (-4) * pybamm.exp(109.451 * sto - 100.006)
    )

    du_dT = (
        -800 + 779 * u_eq - 284 * u_eq**2 + 46 * u_eq**3 - 2.8 * u_eq**4
    ) * 10 ** (-3)

    return du_dT

def electrolyte_diffusivity_PeymanMPM(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration. The original data
    is from [1]. The fit from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte diffusivity
    """

    D_c_e = 5.2 * 10 ** (-10)
    E_D_e = 37040
    arrhenius = pybamm.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def electrolyte_conductivity_PeymanMPM(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration. The original
    data is from [1]. The fit is from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte conductivity
    """

    sigma_e = 1.3
    E_k_e = 34700
    arrhenius = pybamm.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius



def graphite_ocp_PeymanMPM(sto):
    """
    Graphite Open-circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].

    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)
    """

    u_eq = (
        0.063
        + 0.8 * pybamm.exp(-75 * (sto + 0.001))
        - 0.0120 * pybamm.tanh((sto - 0.127) / 0.016)
        - 0.0118 * pybamm.tanh((sto - 0.155) / 0.016)
        - 0.0035 * pybamm.tanh((sto - 0.220) / 0.020)
        - 0.0095 * pybamm.tanh((sto - 0.190) / 0.013)
        - 0.0145 * pybamm.tanh((sto - 0.490) / 0.020)
        - 0.0800 * pybamm.tanh((sto - 1.030) / 0.055)
    )

    return u_eq

def NMC_ocp_PeymanMPM(sto):
    """
    Nickel Managanese Cobalt Oxide (NMC) Open-circuit Potential (OCP) as a
    function of the stochiometry. The fit is taken from Peyman MPM.

    References
    ----------
    Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * (sto**2)
        - 2.0843 * (sto**3)
        + 3.5146 * (sto**4)
        - 2.2166 * (sto**5)
        - 0.5623e-4 * pybamm.exp(109.451 * sto - 100.006)
    )

    return u_eq
def plating_exchange_current_density_OKane2020(c_e, c_Li, T):
    k_plating_0 = pybamm.Parameter("Lithium plating kinetic rate constant [m.s-1]")
#     Ea_k = pybamm.Parameter("Lithium plating kinetic rate constant activation energy")
    Ea_k = 0
    k_plating = k_plating_0 * pybamm.exp(Ea_k/ pybamm.constants.R * (1 / T - 1 / 298.15))
    return pybamm.constants.F * k_plating * c_e


def plating_exchange_current_density_2(c_e, c_Li, T):
    k_plating_0 = pybamm.Parameter("Lithium plating kinetic rate constant (nonlinear) [m.s-1]")
#     Ea_k = pybamm.Parameter("Lithium plating kinetic rate constant activation energy")
    Ea_k = 0
    k_plating = k_plating_0 * pybamm.exp(Ea_k/ pybamm.constants.R * (1 / T - 1 / 298.15))
    return pybamm.constants.F * k_plating * c_e

def SEI_limited_dead_lithium_OKane2022(L_sei):
    """
    Decay rate for dead lithium formation [s-1].
    References
    ----------
    .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
    Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
    and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
    Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
    Parameters
    ----------
    L_sei : :class:`pybamm.Symbol`
        Total SEI thickness [m]
    Returns
    -------
    :class:`pybamm.Symbol`
        Dead lithium decay rate [s-1]
    """

    gamma_0 = pybamm.Parameter("Dead lithium decay constant [s-1]")
    L_inner_0 = pybamm.Parameter("Initial inner SEI thickness [m]")
    L_outer_0 = pybamm.Parameter("Initial outer SEI thickness [m]")
    L_sei_0 = L_inner_0 + L_outer_0

    gamma = gamma_0 * L_sei_0 / L_sei

    return gamma

# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for an LG M50 cell, from the paper

        Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diego Alonso-Alvarez, Robert
        Timms, Valentin Sulzer, Jacqueline Sophie Edge, Billy Wu, Gregory J. Offer, and
        Monica Marinescu. Lithium-ion battery degradation: how to model it. Phys. Chem.
        Chem. Phys., 24:7909-7922, 2022. URL: http://dx.doi.org/10.1039/D2CP00417H,
        doi:10.1039/D2CP00417H.


    based on the paper

        Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
        Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques for
        Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
        Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.


    and references therein.


    Note: the SEI and plating parameters do not claim to be representative of the true
    parameter values. These are merely the parameter values that were used in the
    referenced papers.
    """

    return {
        "chemistry": "lithium_ion",
        # lithium plating (updated in notebook files)
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
        "Lithium plating kinetic rate constant [m.s-1]": 0, # Jeong2024
        "Exchange-current density for plating [A.m-2]": plating_exchange_current_density_OKane2020,
        "Exchange-current density for stripping [A.m-2]": 0,
        "Lithium plating kinetic rate constant (nonlinear) [m.s-1]": 7.52e-11, # Jeong2024
        "Exchange-current density for plating (nonlinear) [A.m-2]": plating_exchange_current_density_2,
        "Initial plated lithium concentration [mol.m-3]": 0.0,
        "Typical plated lithium concentration [mol.m-3]": 1000.0,
        "Lithium plating transfer coefficient": 0.5,
        "Lithium plating tanh stretch [V-1]": 50, # Jeong2024
        "Lithium plating tanh shift [V]": -0.036, # Jeong2024
        "Dead lithium decay constant [s-1]": 1e-06,
        "Dead lithium decay rate [s-1]": SEI_limited_dead_lithium_OKane2022,
        # sei (updated in notebook files)
        "Ratio of lithium moles to SEI moles": 2.0,
        "Inner SEI reaction proportion": 0.0,
        "Inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Outer SEI partial molar volume [m3.mol-1]": 7.545e-05, # Collath2024
        "SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "SEI resistivity [Ohm.m]": 80000.0, # Collath2024
        "Outer SEI solvent diffusivity [m2.s-1]": 2.5000000000000002e-22,
        "Bulk solvent concentration [mol.m-3]": 2636.0,
        "Inner SEI open-circuit potential [V]": 0.1,
        "Outer SEI open-circuit potential [V]": 0.8,
        "Inner SEI electron conductivity [S.m-1]": 8.95e-14,
        "Inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Initial inner SEI thickness [m]": 0.0,
        # "Initial outer SEI thickness [m]": 1.875e-08,
        "Initial outer SEI thickness [m]": 1e-9,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "EC diffusivity [m2.s-1]": 6e-20, # Jeong2024
        "SEI on cracks scaling factor": 0.260, # Jeong2024
        "SEI kinetic rate constant [m.s-1]": 1e-16, # Jeong2024
        "SEI open-circuit potential [V]": 0.4,
        "SEI growth activation energy [J.mol-1]": 30000.0, # Jeong2024
        "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # cell
        "Negative current collector thickness [m]": 2.5e-05, # Mohtat2020
        "Negative electrode thickness [m]": 6.2e-05, # Mohtat2020
        "Separator thickness [m]": 1.2e-05,  # Mohtat2020
        "Positive electrode thickness [m]": 6.7e-05,  # Mohtat2020
        "Positive current collector thickness [m]": 2.5e-05, # Mohtat2020
        "Electrode height [m]": 1.0,  # Mohtat2020
        "Electrode width [m]": 0.205,  # Mohtat2020
        "Cell cooling surface area [m2]": 0.41, # Mohtat2020 (0.205*2)
        "Cell volume [m3]": 3.92e-05,  # Mohtat2020
        "Negative current collector conductivity [S.m-1]": 59600000.0, # Mohtat2020
        "Positive current collector conductivity [S.m-1]": 35500000.0, # Mohtat2020
        "Negative current collector density [kg.m-3]": 8954.0, # Mohtat2020
        "Positive current collector density [kg.m-3]": 2707.0, # Mohtat2020
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385, # Mohtat2024
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897, # Mohtat2020
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0, # Mohtat2020
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0, # Mohtat2020
        "Cell thermal expansion coefficient [m.K-1]": 1.48e-06, #Mohtat2020
        "Nominal cell capacity [A.h]": 5.0,  # Mohtat2020
        "Current function [A]": 5.0,  # Mohtat2020
        "Contact resistance [Ohm]": 0.0, # only activates with model option "contact resistance": "true", but does not solve with current PyBaMM version
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 100.0,  # Mohtat2020
        "Maximum concentration in negative electrode [mol.m-3]": 28746.0,  # Mohtat2020
        "Negative electrode diffusivity [m2.s-1]": graphite_diffusivity_Jeong,  # Mohtat2020
        "Negative electrode OCP [V]": graphite_ocp_PeymanMPM, # Mohtat2020
        "Negative electrode porosity": 0.3,  # Mohtat2020
        "Negative electrode active material volume fraction": 0.6333,  # Jeong2024
        "Negative particle radius [m]": 2.5e-06,  # Mohtat2020
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,  # Mohtat2020
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,  # Mohtat2020
        "Negative electrode transport efficiency": 0.16, # Mohtat2020
        "Negative electrode reference exchange-current density [A.m-2(m3.mol)1.5]"
        "": 1.061e-06, # Mohtat2020
        "Negative electrode charge transfer coefficient": 0.5,  # Mohtat2020
        "Negative electrode double-layer capacity [F.m-2]": 0.2, # Mohtat2020
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_electrolyte_exchange_current_density_Jeong,   # Mohtat2020
        "Negative electrode density [kg.m-3]": 3100.0,  # Mohtat2020
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 1484.2,  # Mohtat2020
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7, # Mohtat2020
        "Negative electrode OCP entropic change [V.K-1]":0,
        # "": graphite_entropic_change_PeymanMPM,
        # negative electrode cracking parameters (from OKane2022 dataset - updated in notebook files)
        "Negative electrode Poisson's ratio": 0.3,
        "Negative electrode Young's modulus [Pa]": 15000000000.0,
        "Negative electrode reference concentration for free of deformation [mol.m-3]": 0.0,
        "Negative electrode partial molar volume [m3.mol-1]": 7e-6, # Pannala2024
        "Negative electrode volume change": graphite_volume_change_mohtat,
        "Negative electrode volume change derivative": graphite_volume_change_mohtat_derivative,
        "Negative electrode Paris' law constant b": 1.12,
        "Negative electrode Paris' law constant m": 2.2,
        "Negative electrode LAM constant proportional term [s-1]": 1.194e-6, # Jeong2024
        "Negative electrode LAM constant exponential term": 1.0,
        "Negative electrode LAM constant proportional term (current-driven) [s-1]": 4.482e-8, # Jeong2024
        "Negative electrode LAM constant exponential term (current-driven)": 1.506, # Jeong2024
        "Negative electrode critical stress [Pa]": 60000000.0,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 100.0,  # Mohtat2020
        "Maximum concentration in positive electrode [mol.m-3]": 35380.0,  # Mohtat2020
        "Positive electrode diffusivity [m2.s-1]": NMC_diffusivity_Jeong, # Mohtat2020
        "Positive electrode OCP [V]": NMC_ocp_PeymanMPM, # Mohtat2020
        "Positive electrode porosity": 0.3,  # Mohtat2020
        "Positive electrode active material volume fraction": 0.4485,  # Jeong2024
        "Positive particle radius [m]": 3.5e-06,  # Mohtat2020
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,  # Mohtat2020
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,  # Mohtat2020
        "Positive electrode transport efficiency": 0.16, # Mohtat2020
        "Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]"
        "": 4.824e-06,   # Mohtat2020
        "Positive electrode charge transfer coefficient": 0.5,  # Mohtat2020
        "Positive electrode double-layer capacity [F.m-2]": 0.2, # Mohtat2020
        "Positive electrode exchange-current density [A.m-2]"
        "": NMC_electrolyte_exchange_current_density_Jeong, # Mohtat2020
        "Positive electrode density [kg.m-3]": 3100.0, # Mohtat2020
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 1491.4, # Jeong2024
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1, # Mohtat2020
        "Positive electrode OCP entropic change [V.K-1]":0,
        # "": NMC_entropic_change_PeymanMPM,
        # positive electrode cracking parameters (from OKane2022 dataset - updated in config files)
        "Positive electrode Poisson's ratio": 0.2,
        "Positive electrode Young's modulus [Pa]": 375000000000.0,
        "Positive electrode reference concentration for free of deformation [mol.m-3]": 0.0,
        "Positive electrode partial molar volume [m3.mol-1]": 7.28e-7, # Pannala2024
        "Positive electrode volume change": nmc_volume_change_mohtat,
        "Positive electrode volume change derivative": 0,
        "Positive electrode Paris' law constant b": 1.12,
        "Positive electrode Paris' law constant m": 2.2,
        "Positive electrode LAM constant proportional term [s-1]": 3.759e-7, # Jeong2024
        "Positive electrode LAM constant exponential term": 1.0,
        "Positive electrode LAM constant proportional term (current-driven) [s-1]": 5.499e-9, # Jeong2024
        "Positive electrode LAM constant exponential term (current-driven)": 1.77474, # Jeong2024
        "Positive electrode critical stress [Pa]": 375000000.0,
        # separator
        "Separator porosity": 0.4, # Mohtat2020
        "Separator Bruggeman coefficient (electrolyte)": 1.5, # Mohtat2020
        "Separator density [kg.m-3]": 397.0, # Mohtat2020
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0, # Mohtat2020
        "Separator thermal conductivity [W.m-1.K-1]": 0.16, # Mohtat2020
        "Separator transport efficiency ": 0.25, # Mohtat2020
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0, # Mohtat2020
        "Cation transference number": 0.38,  # Mohtat2020
        "Thermodynamic factor": 1.34, # Mohtat2020
        "Typical lithium ion diffusivity [m2.s-1]": 5.34e-10, # Mohtat2020
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_PeymanMPM, # Mohtat2020
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_PeymanMPM, # Mohtat2020
        # experiment
        "Reference temperature [K]": 298.15,
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0, # Mohtat2020
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0, # Mohtat2020
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0.0, # Mohtat2020
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0.0, # Mohtat2020
        "Edge heat transfer coefficient [W.m-2.K-1]": 5.0, # Mohtat2020
        "Total heat transfer coefficient [W.m-2.K-1]": 0.3123, # Jeong2024
        "Ambient temperature [K]": 298.15, # Mohtat2020
        "Number of electrodes connected in parallel to make a cell": 1.0, # Mohtat2020
        "Number of cells connected in series to make a battery": 1.0, # Mohtat2020
        "Lower voltage cut-off [V]": 3.0,
        "Upper voltage cut-off [V]": 4.2, # Mohtat2020
        "Open-circuit voltage at 0% SOC [V]": 3.0,
        "Open-circuit voltage at 100% SOC [V]": 4.2, # Mohtat2020
        # "Initial stoichiometry in negative electrode": 0.0017, # Mohtat2020
        # "Initial stoichiometry in positive electrode": 0.8907, # Mohtat2020
        "Initial stoichiometry in negative electrode": 0.006, # Jeong2024
        "Initial stoichiometry in positive electrode": 0.874, # Jeong2024
        "Initial concentration in negative electrode [mol.m-3]": pybamm.Parameter("Initial stoichiometry in negative electrode")*pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]"), # Jeong2024
        "Initial concentration in positive electrode [mol.m-3]": pybamm.Parameter("Initial stoichiometry in positive electrode")*pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]"), # Jeong2024
        "Initial temperature [K]": 298.15, # Mohtat2020
        "citations": ["Jeong2024"],
    }
