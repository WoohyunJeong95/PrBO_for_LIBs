import pybamm
import numpy as np
from scipy import interpolate
from datetime import datetime, timedelta

temperature_C = 25
R_const = 8.314

n_days = 100
dch_per_one_day = 10
one_day_time = 24


param = pybamm.ParameterValues("Jeong2024")
options = {"thermal":"lumped","SEI": "ec reaction limited","SEI porosity change":"true","loss of active material":"current and stress-driven","particle mechanics":"swelling only","lithium plating": "irreversible", "lithium plating porosity change": "true"}
model = pybamm.lithium_ion.SPM(options=options)
param.update(
    {"Initial temperature [K]": 273.15 + temperature_C,
     "Ambient temperature [K]": 273.15 + temperature_C,
     })

# Initial capacity
capa_0 = 4.910019471679001

def aging_test_results(I1,I2,I3,V1,V2):
    s = pybamm.step.string
    solver = pybamm.CasadiSolver(mode='safe', return_solution_if_failed_early=True)
    n_exp_per_day_arr = []
    for c in range(n_days):
        st_time_1 = datetime(2024, 1, 1, 0, 0) + timedelta(hours=c * one_day_time)

        experiment1 = pybamm.Experiment([
            (s(f"Charge at {I1}C until {V1}V", start_time=st_time_1),
             s(f"Charge at {I2}C until {V2}V"),
             s(f"Charge at {I3}C until {4.2}V"),
             "Discharge at 0.1C until 3.0V",)])
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment1, solver=solver)
        if c == 0:
            solution = sim.solve(initial_soc=0)
        else:
            solution = sim.solve(starting_solution=solution)

        dch_time_exp1 = solution.cycles[-1].steps[-1]['Time [h]'].entries[-1] - \
                        solution.cycles[-1].steps[-1]['Time [h]'].entries[0]

        if dch_time_exp1 >= dch_per_one_day:
            experiment2 = pybamm.Experiment([
                (s("Rest for 8 hours", start_time=st_time_2))])
            sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment2, solver=solver)
            solution = sim.solve(starting_solution=solution)
            n_exp_day = 2
            n_exp_per_day_arr.append(n_exp_day)


        else:
            dch_time_left = dch_per_one_day - dch_time_exp1
            experiment2 = pybamm.Experiment([
                (f"Charge at {I1}C until {V1}V",
                 f"Charge at {I2}C until {V2}V",
                 f"Charge at {I3}C until {4.2}V",
                 f"Discharge at 0.1C for {dch_time_left} hours or until 3.0V",)])
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
                    (s(f"Rest for {rest_time} hours", start_time=st_time_2))])
                sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment3, solver=solver)
                solution = sim.solve(starting_solution=solution)
                n_exp_day = 3
                n_exp_per_day_arr.append(n_exp_day)

    # Qloss
    Q_loss_arr = capa_0 - solution.summary_variables['Capacity [A.h]']
    Q_loss_arr = np.insert(Q_loss_arr, 0, 0)

    # Capacity retention
    cap_ret = (capa_0-Q_loss_arr)/capa_0

    # Charge time
    start_idx = n_exp_per_day_arr[0]
    ch_time = solution.cycles[start_idx].steps[-1]['Time [h]'].entries[0] - \
              solution.cycles[start_idx]['Time [h]'].entries[0]
    for s in range(n_exp_per_day_arr[1] - 2):
        ch_time += solution.cycles[start_idx + 1 + s].steps[-1]['Time [h]'].entries[0] - \
                   solution.cycles[start_idx + 1 + s]['Time [h]'].entries[0]


    return [ch_time, cap_ret[-1]]