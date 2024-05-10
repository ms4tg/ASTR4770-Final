#  2D Hydrocode Visualization

## Project Overview

This project aims to create cool video visualizations of two interesting instabilities arising in 2D hydrocode simulations. The two instabilities explored are:

1. **Kelvin-Helmholtz Instability**
2. **Rayleigh-Taylor Instability**

Due to its limited scope, the project focuses specifically on these two instabilities and lacks the functionality to implement more complicated simulations.

## Main Components

### `main.ipynb`

This Jupyter notebook serves as the main interface for visualization. It sets up the parameters and creates video simulations of the processes. You can interact with the parameters and run the simulations directly from this notebook.

### `utils.py`

This module contains all the utility functions needed for computation, including:

- Conversion between primitive and conservative variables
- Calculation of gradients
- Extrapolation
- Addition of ghost cells
- Addition of source terms due to gravity

### `rusanov_flux.py`

This module contains the functions necessary to approximate the flux using the Rusanov flux method, which is crucial for simulating the fluid dynamics.

### 'test1.mp4', 'test2.mp4'

Those two mp4 files are sample video simulations generated for the two tests mentioned above. 
