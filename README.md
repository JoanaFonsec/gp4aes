
###### SHORT STORY
- Run this DEMO:
- `python scripts/DEMO.py datasets/finland_forecast_14_04_17_04.nc datasets/finland_ocean_color_14_04_17_04.nc config/finland_lres.json config/finland_hres.json results/finland_complete_mission.h5`


###### LONG STORY   
## 1 - Read .nc datasets into .h5 files and plot them
- LR data:
    - `python scripts/read_nc_into_h5_file.py datasets/finland_forecast_14_04_17_04.nc config/finland_lres.json results/finland_lres.h5 1618610399`
- HR data:
    - `python scripts/read_nc_into_h5_file.py datasets/finland_ocean_color_14_04_17_04.nc config/finland_hres.json results/finland_hres.h5 1618610399`

## 2 - Train kernel with data from previous days
Kernel is trained with LR data from the previous 3 days, and the parameters are printed in the terminal. Since operational area contains NaNs, it is necessary to choose a clipped area from the original, containing only ocean samples. Runs a multi-thread optimizer and can take up to 5/7 min:
- `python scripts/train_kernel_GP.py results/finland_lres.h5 1618610399 MAT 3 --clipped_area 61.412 61.80 20.771 21.31`

For reference, my last results were: {44.29588721 0.54654887 0.26656638}

## 3 - Prediction with scattered data
Predict HR data with Kernel trained as aforementioned. Measurements are scattered. To change datasets size, hardcode them on the script.
- `python gpr/gpr_interp.py out/finland_hres_oceancolor.h5 1618610399 MAT --kernel_params 53.24932783 0.48764177 0.2128522`

Run script that plots ground truth, predicted mean, relative error and standard deviation:
-  `python util/plot_gp.py out/finland_hres_oceancolor.h5`

Remark: _This step is not needed to run the trajectory script_.

## 4 - Compute trajectory
Compute trajectory based on GP estimation algorithm:
- `python scripts/run_mission.py results/finland_hres_oceancolor.h5 results/traj_finland_hres_oceancolor.h5 1618610399 MAT --kernel_params 44.29588721 0.54654887 0.26656638`

To plot the trajectory and KPIs:
- `python util/plot_traj.py out/traj_finland_hres_oceancolor.h5 --ref --grad_error`

## 5 - Data Assimilation
Predict chl _a_ concentration with LR dataset:
- `python gpr/gpr_interp.py out/finland_lres_forecast.h5 1618653600 MAT --kernel_params 44.29588721 0.54654887 0.26656638`

Compute average relative error compared to ground truth:
- `python util/compare_data.py out/finland_hres_oceancolor.h5 out/finland_lres_forecast.h5`

Compute trajectory without measurement error (tuning of some parameters has to be hardcoded):
-`python control/sim_gpr.py out/finland_hres_oceancolor.h5 out/traj_finland_hres_oceancolor.h5 1618610399 MAT --std 0.00001`

Predict chl _a_ concentration with LR dataset together with the computed trajectory:
- If a fixed standard deviation is to be used for scattered data:
    - `python gpr_tests/gpr_augmented.py out/finland_lres_forecast.h5 out/traj_finland_hres_oceancolor.h5 1618653600 MAT --kernel_params 44.29588721 0.54654887 0.26656638`
- Conversely, if the standard deviation of the scattered data is to be linear with the distance to the trajectory:
    -`python gpr_tests/gpr_augmented.py out/finland_lres_forecast.h5 out/traj_finland_hres_oceancolor.h5 1618653600 MAT --kernel_params 44.29588721 0.54654887 0.26656638 --cont_std`

Repeat average relative error computations:
- `python util/compare_data.py out/finland_hres_oceancolor.h5 out/finland_lres_forecast.h5`

For visual inspection, plot the predicted mean:
- `python util/plot_gp.py out/finland_lres_forecast.h5`
