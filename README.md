#  GP4AES - Gaussian Processes for Adaptive Environmental Sampling
This package implements an algorithm for Gaussian Regression-based Adaptive Sampling for environmental variables, in particular, those found in water bodies. 

## Installation
To install the package, make sure to have installed Python Poetry in your system. Then, run the following commands:

1. Install the `gp4aes` library:
```
poetry install
```

2. Run the demo script:
```
poetry run python scripts/DEMO.py datasets/finland_forecast_14_04_17_04.nc datasets/finland_ocean_color_14_04_17_04.nc config/finland_lres.json config/finland_hres.json results/finland_complete_mission.h5
```

## Try each Submodule
Test each submodule individually.
### Read .nc datasets into .h5 files and plot them

1. Low Resolution data:
```
poetry run python scripts/read_nc_into_h5_file.py datasets/finland_forecast_14_04_17_04.nc config/finland_lres.json results/finland_lres.h5 1618610399
```
2. High Resolution data:
```
poetry run python scripts/read_nc_into_h5_file.py datasets/finland_ocean_color_14_04_17_04.nc config/finland_hres.json results/finland_hres.h5 1618610399
```

### Train kernel with data from previous days
Kernel is trained with LR data from the previous 3 days, and the parameters are printed in the terminal. Since operational area contains NaNs, it is necessary to choose a clipped area from the original, containing only ocean samples. Runs a multi-thread optimizer and can take up to 5/7 min:
```
poetry run python scripts/train_kernel_GP.py results/finland_lres.h5 1618610399 MAT 3 --clipped_area 61.412 61.80 20.771 21.31
```

A sample minimization iteration is the following parameters:  `44.29588721 0.54654887 0.26656638`

### Prediction with scattered data
Predict HR data with Kernel trained as aforementioned. Measurements are scattered. To change datasets size, hardcode them on the script.
```
poetry run python scripts/gpr_interp.py results/finland_hres.h5 1618610399 MAT --kernel_params 53.24932783 0.48764177 0.2128522
```

Run script that plots ground truth, predicted mean, relative error and standard deviation:
```
poetry run python util/plot_gp.py results/finland_hres.h5
```

Remark: _This step is not needed to run the trajectory script_.

### Compute trajectory
Compute trajectory based on the Gaussian Process estimation algorithm:
```
poetry run python scripts/run_mission.py results/finland_hres.h5 results/finland_complete_mission.h5 1618610399 MAT --kernel_params 44.29588721 0.54654887 0.26656638
```

To plot the results after the mission:
```
poetry run python scripts/plot_results_after_mission.py results/finland_complete_mission.h5
```