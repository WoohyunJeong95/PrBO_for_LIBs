# PrBO_for_LIBs
This repository contains modeling results for two battery models with different fidelity levels, employed in the study 
**"Health-Conscious Charging of Lithium-ion Batteries Using Bayesian Optimization Guided by a Semi-Empirical Aging Model"**
Additionally, it includes the results of prior knowledge-guided Bayesian optimization (PrBO) method applied to optimize charging strategies based on these models. 

## Installation
1. Create a new virtual environment and install PyBaMM.
   - This project uses 'pybamm==24.9.0'.
   - Some equations in the degradation model ('.py') files have been modified.
2. Install the custom parameter set used in this work.
   - pip install -e pybamm-parameter-set-Jeong2024

## Project structure
1. EM testbed
   - Capacity degradation validation results using the electrochemical-thermal-aging model.
2. ECM+aging model
   - Validation results of equivalent circuit model (ECM).
   - Validataion results of semi-empirical aging models (calendar & cycling degradation).
3. Model runs
   - Simulation examples of the EM testbed model and ECM+aging model, which allow users to modify input parameters.
   - Grid samples used for constructing the ECM+aging interpolation model, used as a prior knowledge in Bayesian optimization.
4. PrBO
   - Initial dataset obtained using the ECM+aging model.
   - Prior knowledge-guided Bayesian optimization to efficiently identify optimal charging strategies.
 
   
