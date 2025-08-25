import pybamm
import numpy as np
from scipy import interpolate
from datetime import datetime, timedelta
import pickle
import os



temperature_C = 25
n_days = 100
one_day_time = 24
R_const = 8.314
dch_per_one_day = 10

# Initial capacity
capa_0 = 4.910019471679001

with open('ECM_cal_B.pkl', 'rb') as f:
    B_cal = pickle.load(f)
with open('ECM_cal_Ea.pkl', 'rb') as f:
    Ea_cal = pickle.load(f)
with open('ECM_cal_z.pkl', 'rb') as f:
    z_cal = pickle.load(f)

with open('ECM_cyc_B.pkl', 'rb') as f:
    B_cyc = pickle.load(f)
with open('ECM_cyc_Ea.pkl', 'rb') as f:
    Ea_cyc = pickle.load(f)
with open('ECM_cyc_z.pkl', 'rb') as f:
    z_cyc = pickle.load(f)


z_corr_cyc = 0
z_corr_cal = 0

def dQdt_cal(SOC,T,Qcal):
    B = B_cal(SOC)
    exp_term = np.exp(-Ea_cal/(R_const*T))
    dQdt = z_cal*B*exp_term*(Qcal/(B*exp_term))**(1-1/z_cal)
    return dQdt

def dQdt_cyc(I,T,Qcyc):
    c_rate = abs(I)/5
    B = B_cyc(c_rate)
    exp_term = np.exp(Ea_cyc/(R_const*T))
    dQdt = abs(I)/3600*z_cyc*B*exp_term*(Qcyc/(B*exp_term))**(1-1/z_cyc)
    return dQdt

model = pybamm.equivalent_circuit.Thevenin(options={"operating mode":"current","number of rc elements":0})
param = model.default_parameter_values
path = os.getcwd()
ocv_data = pybamm.parameters.process_1D_data("ecm_ocv.csv", path=path)
r0_data = pybamm.parameters.process_3D_data_csv("ecm_r0.csv", path=path)
def r0(T_cell, current, soc):
    name, (x, y) = r0_data
    return pybamm.Interpolant(x, y, [T_cell, current, soc], name)
param.update(
{   'R0 [Ohm]': r0,}
)
param.update(
    {   "Upper voltage cut-off [V]":4.2,
        "Cell-jig heat transfer coefficient [W/K]": 0.1601733,
        "Cell thermal mass [J/K]": 173.6341749,
        "Jig thermal mass [J/K]": 1e20,
        "Jig-air heat transfer coefficient [W/K]": 0,
        "Cell capacity [A.h]": 5,
        "Nominal cell capacity [A.h]": 5,
        "Initial SoC": 0.001,
        "Lower voltage cut-off [V]": 3.0,
        "Open-circuit voltage [V]": ocv_data,
        "Entropic change [V/K]": 0,
        "Initial temperature [K]": temperature_C + 273.15,
        "Ambient temperature [K]": temperature_C + 273.15,
        "Current function [A]": 0,
    }
)


def aging_test_results(I1,I2,I3,V1,V2):
    s = pybamm.step.string
    n_exp_per_day_arr = []
    # Multi-simulation ########################################################################
    solver = pybamm.CasadiSolver(mode='safe', return_solution_if_failed_early=True)
    for c in range(2):
        st_time_1 = datetime(2024, 1, 1, 0, 0) + timedelta(hours=c * one_day_time)

        experiment1 = pybamm.Experiment([
            (s(f"Charge at {I1}C until {V1}V", start_time=st_time_1, period='5 minutes'),
             s(f"Charge at {I2}C until {V2}V", period='5 minutes'),
             s(f"Charge at {I3}C until {4.2}V", period='5 minutes'),
             s("Discharge at 0.1C until 3.0V", period='5 minutes'),
             )])
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment1, solver=solver)
        if c == 0:
            solution = sim.solve()
        else:
            solution = sim.solve(starting_solution=solution)

        dch_time_exp1 = solution.cycles[-1].steps[-1]['Time [h]'].entries[-1] - \
                        solution.cycles[-1].steps[-1]['Time [h]'].entries[0]

        if dch_time_exp1 >= dch_per_one_day:
            experiment2 = pybamm.Experiment([
                (s("Rest for 8 hours", start_time=st_time_2, period='10 minutes'))])
            sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment2, solver=solver)
            solution = sim.solve(starting_solution=solution)
            n_exp_day = 2
            n_exp_per_day_arr.append(n_exp_day)


        else:
            dch_time_left = dch_per_one_day - dch_time_exp1
            experiment2 = pybamm.Experiment([
                (s(f"Charge at {I1}C until {V1}V", period='5 minutes'),
                 s(f"Charge at {I2}C until {V2}V", period='5 minutes'),
                 s(f"Charge at {I3}C until {4.2}V", period='5 minutes'),
                 s(f"Discharge at 0.1C for {dch_time_left} hours or until 3.0V", period='5 minutes')
                 )])
            sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment2, solver=solver)
            solution = sim.solve(starting_solution=solution)
            dch_time_exp2 = solution.cycles[-1].steps[-1]['Time [h]'].entries[-1] - \
                            solution.cycles[-1].steps[-1]['Time [h]'].entries[0]

            st_time_2_hr = solution.cycles[-1].steps[-1]['Time [h]'].entries[-1] - \
                           solution.cycles[-2]['Time [h]'].entries[0]
            st_time_2_h = int(np.floor(st_time_2_hr))
            st_time_2_m = int(np.floor((st_time_2_hr - st_time_2_h) * 60))
            st_time_2_s = int(np.ceil(((st_time_2_hr - st_time_2_h) * 60 - st_time_2_m) * 60))
            if st_time_2_s == 60:
                st_time_2_m = st_time_2_m + 1
                st_time_2_s = 0
            if st_time_2_m == 60:
                st_time_2_h = st_time_2_h + 1
                st_time_2_m = 0
            st_time_2 = datetime(2024, 1, 1, st_time_2_h, st_time_2_m, st_time_2_s) + timedelta(hours=c * one_day_time)
            rest_time = 24 - (st_time_2_h + st_time_2_m / 60 + st_time_2_s / 3600)
            if abs(dch_time_exp1 + dch_time_exp2 - dch_per_one_day) < 1e-5:
                experiment3 = pybamm.Experiment([
                    (s(f"Rest for {rest_time} hours", start_time=st_time_2, period='10 minutes'))])
                sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment3, solver=solver)
                solution = sim.solve(starting_solution=solution)
                n_exp_day = 3
                n_exp_per_day_arr.append(n_exp_day)
    ##########################################################################################

    start_idx = n_exp_per_day_arr[0]
    sol_time = solution.cycles[start_idx]['Time [h]'].entries
    sol_current = solution.cycles[start_idx]['Current [A]'].entries
    sol_temp = solution.cycles[start_idx]['Cell temperature [K]'].entries
    sol_SOC = solution.cycles[start_idx]['SoC'].entries

    for s in range(1, n_exp_per_day_arr[1]):  # 2nd day data concatenate
        sol_time = np.concatenate([sol_time, solution.cycles[start_idx + s]['Time [h]'].entries])
        sol_current = np.concatenate([sol_current, solution.cycles[start_idx + s]['Current [A]'].entries])
        sol_temp = np.concatenate([sol_temp, solution.cycles[start_idx + s]['Cell temperature [K]'].entries])
        sol_SOC = np.concatenate([sol_SOC, solution.cycles[start_idx + s]['SoC'].entries])
    sol_time = sol_time - solution.cycles[start_idx]['Time [h]'].entries[0] # starts from 0

    # Cycling aging & Calendar aging
    Q_loss_cyc_corr = np.full(n_days + 1, 1e-10)
    time_h = sol_time
    Temp = sol_temp
    current_arr = sol_current
    delta_time_s = np.diff(time_h) * 3600

    Q_loss_cal = np.full(n_days + 1, 1e-10)
    sol_time_diff_s = np.diff(sol_time) * 3600

    for c in range(n_days):
        Q_loss_cyc_temp = np.full(len(time_h), Q_loss_cyc_corr[c])
        Q_loss_cal_temp = np.full(len(sol_time), Q_loss_cal[c])
        for t in range(len(time_h) - 1):
            corr_term = ((5 - (Q_loss_cyc_temp[t] + Q_loss_cal_temp[t])) / 5)
            dQdt = dQdt_cyc(current_arr[t], Temp[t], Q_loss_cyc_temp[t]) * corr_term ** z_corr_cyc
            Q_loss_cyc_temp[t + 1] = Q_loss_cyc_temp[t] + dQdt * delta_time_s[t]
            dQdt = dQdt_cal(sol_SOC[t], sol_temp[t], Q_loss_cal_temp[t]) * corr_term ** z_corr_cal
            Q_loss_cal_temp[t + 1] = (Q_loss_cal_temp[t] + dQdt * sol_time_diff_s[t])

        Q_loss_cyc_corr[c + 1] = Q_loss_cyc_temp[-1]
        Q_loss_cal[c + 1] = (Q_loss_cal_temp[-1])

    Q_loss_tot = Q_loss_cyc_corr + Q_loss_cal
    cap_ret = (capa_0 - Q_loss_tot) / capa_0

    # Charge time
    ch_time = solution.cycles[start_idx].steps[-1]['Time [h]'].entries[0] - \
              solution.cycles[start_idx]['Time [h]'].entries[0]
    for s in range(n_exp_per_day_arr[1] - 2):
        ch_time += solution.cycles[start_idx + 1 + s].steps[-1]['Time [h]'].entries[0] - \
                   solution.cycles[start_idx + 1 + s]['Time [h]'].entries[0]

    return [ch_time, cap_ret[-1]]