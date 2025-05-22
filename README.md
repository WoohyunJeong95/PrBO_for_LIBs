# PrBO_for_LIBs
This repository contains modeling results for two battery models with different fidelity levels, employed in the study 
**"Health-Conscious Charging of Lithium-ion Batteries Using Bayesian Optimization Guided by a Semi-Empirical Aging Model"**
The models aim to capture the degradation behavior of lithium-ion batteries under various charging conditions while maintaining a balance between accuracy and computational efficiency.

# Installation
1. Create a new virtual environment and install PyBaMM.
   - This project uses 'pybamm==24.9.0'.
   - Some equations in the degradation model ('.py') files have been modified.
2. Install the custom parameter set used in this work:
   '''bash
   pip install -e pybamm-parameter-set-Jeong2024

  
