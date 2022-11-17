


## 1 - Interpolate
- LR data:
    - `./interpolation/interp_daily_forecast.py data/finland_forecast_14_04_17_04.nc config/finland_lres_forecast.json out/finland_lres_forecast.h5`
- HR data:
    - `./interpolation/interp_daily_forecast.py data/finland_ocean_color_14_04_17_04.nc config/finland_hres_oceancolor.json out/finland_hres_oceancolor.h5`
- Plot data on the "prediction" day:
    - LR: `./util/plot_interp.py out/finland_lres_forecast.h5 1618610399`
    - HR: `./util/plot_interp.py out/finland_hres_oceancolor.h5 1618610399`

## 2 - Train kernel with previous days
Kernel is trained with LR data from the previous 3 days, and the parameters are printed in the terminal. Since operational area contains NaNs, it is necessary to choose a clipped area from the original, containing only ocean samples. Runs a multi-thread optimizer and can take up to 5/7 min:
- `./gpr/multitask_learning.py out/finland_lres_forecast.h5 1618610399 MAT 3 --clipped_op_area 61.412 61.80 20.771 21.31`

For reference, my last results were: {44.29588721 0.54654887 0.26656638}

## 3 - Prediction with scattered data
Predict HR data with Kernel trained as aforementioned. Measurements are scattered. To change datasets size, hardcode them on the script.
- `./gpr/gpr_interp.py out/finland_hres_oceancolor.h5 1618610399 MAT --kernel_params 53.24932783 0.48764177 0.2128522`

Run script that plots ground truth, predicted mean, relative error and standard deviation:
-  `./util/plot_gp.py out/finland_hres_oceancolor.h5`

Remark: _This step is not needed to run the trajectory script_.

## 4 - Compute trajectory
Compute trajectory based on GP estimation algorithm:
- `python control/sim_gpr.py out/finland_hres_oceancolor.h5 out/traj_finland_hres_oceancolor.h5 1618610399 MAT --kernel_params 44.29588721 0.54654887 0.26656638`

For repetitive runs, it is advisable to hard code the kernele parameters.

To plot the trajectory and KPIs:
- `python util/plot_traj.py out/traj_finland_hres_oceancolor.h5 --ref --grad_error`

## 5 - Data Assimilation
Predict chl _a_ concentration with LR dataset:
- `./gpr/gpr_interp.py out/finland_lres_forecast.h5 1618653600 MAT --kernel_params 44.29588721 0.54654887 0.26656638`

Compute average relative error compared to ground truth:
- `./util/compare_data.py out/finland_hres_oceancolor.h5 out/finland_lres_forecast.h5`

Compute trajectory without measurement error (tuning of some parameters has to be hardcoded):
-`./control/sim_gpr.py out/finland_hres_oceancolor.h5 out/traj_finland_hres_oceancolor.h5 1618610399 MAT --std 0.00001`

Predict chl _a_ concentration with LR dataset together with the computed trajectory:
- If a fixed standard deviation is to be used for scattered data:
    - `./gpr_tests/gpr_augmented.py out/finland_lres_forecast.h5 out/traj_finland_hres_oceancolor.h5 1618653600 MAT --kernel_params 44.29588721 0.54654887 0.26656638`
- Conversely, if the standard deviation of the scattered data is to be linear with the distance to the trajectory:
    -`./gpr_tests/gpr_augmented.py out/finland_lres_forecast.h5 out/traj_finland_hres_oceancolor.h5 1618653600 MAT --kernel_params 44.29588721 0.54654887 0.26656638 --cont_std`

Repeat average relative error computations:
- `./util/compare_data.py out/finland_hres_oceancolor.h5 out/finland_lres_forecast.h5`

For visual inspection, plot the predicted mean:
- `./util/plot_gp.py out/finland_lres_forecast.h5`
