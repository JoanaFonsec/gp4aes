import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

plt.style.reload_library()
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'xtick.labelsize': 20,
                    'ytick.labelsize': 20,
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'legend.fontsize': 20,
                    'legend.frameon' : True
                    })

###################################### Class plotter
class Plotter:
    def __init__(self, position1, gradient1, measurements1, position2, gradient2, measurements2, lon, lat, chl, chl_ref, meas_per, time_step):
        self.lon = lon
        self.lat = lat
        self.lat = lat
        self.chl = chl
        self.chl_ref = chl_ref
        self.meas_per = int(meas_per)

        # Trim zeros and last entry so that all meas/grads are matched
        position1 = position1[:-1,:]
        position1 = position1[~np.all(position1 == 0, axis=1)]
        position2 = position2[:-1,:]
        position2 = position2[~np.all(position2 == 0, axis=1)]
        
        # Get start and end time from arguments
        self.start_time = 0
        self.end_time = time_step*(min(len(gradient1[:, 0]),len(gradient2[:, 0]))-2)/3600.0

        # Determine start and stop indexes for the mission period
        self.idx_start = int(3600.0 / time_step / self.meas_per * self.start_time)
        self.idx_end = int(3600.0 / time_step / self.meas_per * self.end_time)
        self.vector_length = self.idx_end - self.idx_start

        # Create mission time axis for 1
        self.it = np.linspace(self.start_time, self.end_time, self.idx_end-self.idx_start)

        # Adjust all data 1
        self.position1 = position1[meas_per*self.idx_start:meas_per*self.idx_end,:]
        self.measurements1 = measurements1[self.idx_start:self.idx_end]
        self.gradient1 = gradient1[self.idx_start:self.idx_end,:]
        # Adjust all data 2
        self.position2 = position2[meas_per*self.idx_start:meas_per*self.idx_end,:]
        self.measurements2 = measurements2[self.idx_start:self.idx_end]
        self.gradient2 = gradient2[self.idx_start:self.idx_end,:]

################################################################################ PLOT GRADIENT #######################################################
    def gradient_comparison(self):

        # gt => ground truth
        ground_truth1 = np.zeros([self.gradient1.shape[0], 2])
        ground_truth2 = np.zeros([self.gradient1.shape[0], 2])
        error1 = np.zeros(self.gradient1.shape[0])
        error2 = np.zeros(self.gradient1.shape[0])
        avg_error_1 = 0
        avg_error_2 = 0

        # Ground truth gradient everywhere
        ground_truth = np.gradient(self.chl)
        gt_norm = np.sqrt(ground_truth[0]**2 + ground_truth[1]**2)
        everywhere_gradient = (RegularGridInterpolator((self.lon, self.lat), ground_truth[0]/gt_norm),
                    RegularGridInterpolator((self.lon, self.lat), ground_truth[1]/gt_norm))

        # Compute ground truth gradients
        for i in range(self.vector_length):
            x = self.meas_per*i

            ground_truth1[i, 0] = everywhere_gradient[0]((self.position1[x,0], self.position1[x,1]))
            ground_truth1[i, 1] = everywhere_gradient[1]((self.position1[x,0], self.position1[x,1]))
            ground_truth2[i, 0] = everywhere_gradient[0]((self.position2[x,0], self.position2[x,1]))
            ground_truth2[i, 1] = everywhere_gradient[1]((self.position2[x,0], self.position2[x,1]))            
            error1[i] = np.dot(self.gradient1[i], ground_truth1[i]) / (np.linalg.norm(self.gradient1[i]) * np.linalg.norm(ground_truth1[i]))
            error2[i] = np.dot(self.gradient2[i], ground_truth2[i]) / (np.linalg.norm(self.gradient2[i]) * np.linalg.norm(ground_truth2[i]))
        print("Average gradient error for GP: {}".format(np.mean(abs(error1))), "Average gradient error for LSQ: {}".format(np.mean(abs(error2))))

        # Plot gradient angle
        fig, ax = plt.subplots(figsize=(15, 3))
        plt.plot(self.it, error1, 'r-', linewidth=1, label='GP')
        plt.plot(self.it, error2, 'k-', linewidth=1, label='LSQ')
        plt.xlabel('Mission time [h]')
        plt.ylabel('Gradient [rad]')
        plt.legend(loc=4, shadow=True)
        plt.grid(True)
        plt.axis([self.start_time, self.end_time, -1, 1.1])

        return fig

################################################################################ PLOT CHL #######################################################
    def chl_comparison(self):
        
        fig, ax = plt.subplots(figsize=(15, 3))
        plt.plot(self.it, abs(self.measurements1-self.chl_ref), 'r-', linewidth=1, label="GP")
        plt.plot(self.it, abs(self.measurements2-self.chl_ref), 'k-', linewidth=1, label="LSQ")
        plt.xlabel('Mission time [h]')
        plt.ylabel('Concentration\n mm/mm3]')
        plt.axis([self.start_time, self.end_time, 0, 0.2]) # 6,9 and 2,9
        plt.legend(loc=4, shadow=True)
        plt.grid(True)

        print("Average front distance for GP: {}".format(np.mean(abs(self.measurements1-self.chl_ref))), "Average front distance for LSQ: {}".format(np.mean(abs(self.measurements2-self.chl_ref))))

        return fig
