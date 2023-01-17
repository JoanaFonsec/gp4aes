import numpy as np
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import spatial
import geopy.distance

plt.style.reload_library()
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'xtick.labelsize': 20,
                    'ytick.labelsize': 20,
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'legend.fontsize': 20,
                    'legend.frameon' : True
                    })


######################################## For plots with arrows
class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = mpatches.FancyArrow(0, 0.7*height, width, 0, length_includes_head=True, head_width=0.25*height )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


###################################### Class plotter
class Plotter:
    def __init__(self, position, lon, lat, chl, gradient, measurements, chl_ref, meas_per, time_step):
        self.lon = lon
        self.lat = lat
        self.lat = lat
        self.chl = chl
        self.chl_ref = chl_ref
        self.meas_per = meas_per

        # Trim zeros and last entry so that all meas/grads are matched
        position = position[:-1,:]
        position = position[~np.all(position == 0, axis=1)]

        # Get start and end time from arguments
        self.start_time = 0
        self.end_time = time_step*(len(position[:, 0])-1)/3600

        # Determine start and stop indexes for the mission period
        self.idx_start = int(3600 / time_step / self.meas_per * self.start_time)
        self.idx_end = int(3600 / time_step / self.meas_per * self.end_time)
        self.vector_length = self.idx_end - self.idx_start

        # Determine index of traj where AUV reaches the front
        self.idx_trig = 0
        for i in range(len(measurements)):
            if chl_ref - 5e-3 <= measurements[i]:
                self.idx_trig = i
                break

        # Create mission time axis
        if position.shape[1] == 3:  # this one doesn't run
            delta_t = (position[-1,-1] - position[0,-1])/3600
            self.it = np.linspace(0, delta_t, len(measurements))
        elif position.shape[1] == 2:
            self.it = np.linspace(self.start_time, self.end_time, self.idx_end-self.idx_start)
            self.idx_trig_time = self.it[self.idx_trig]

        # Create zoom 1 and 2 time axis
        self.zoom1_start = 15500
        self.zoom1_end = 26000
        self.zoom2_start = 18710
        self.zoom2_end = 19870

        # Adjust all data
        self.position = position[meas_per*self.idx_start:meas_per*self.idx_end,:]
        self.measurements = measurements[self.idx_start:self.idx_end]
        self.gradient = gradient[self.idx_start:self.idx_end,:]

    # Get the chl front line
    def get_front_line(self):
        path = None
        
        # Determine which path is the longest (treat this as the gradient path)
        longest_path = 0
        for i in self.cs.collections[0].get_paths():
            path_length = i.vertices.shape[0]
            if path_length>longest_path:
                longest_path = path_length
                path = i

        longest = path.vertices

        # Upsample the chl front line
        conxyd = np.empty((0,2))
        discretmax = 0.00005

        for ii in range(longest.shape[0]-1):
            vertdist = np.sqrt((longest[ii, 0]-longest[ii+1, 0])**2 + (longest[ii, 1]-longest[ii+1, 1])**2)
            discret = int(vertdist/discretmax)

            if discret < 0.5:
                conxyii = np.empty((0, 2))
            else:
                conxyii = np.hstack((np.linspace(longest[ii, 0], longest[ii+1, 0], discret).reshape(-1,1), np.linspace(longest[ii, 1], longest[ii+1, 1], discret).reshape(-1,1)))

            conxyd = np.vstack((conxyd, conxyii))

        return conxyd

############################################################################################# PLOT TRAJECTORY ###################################################

    def mission_overview(self):

        fig, ax = plt.subplots(figsize=(15, 6))
        xx, yy = np.meshgrid(self.lon, self.lat, indexing='ij')
        p = ax.pcolormesh(xx, yy, self.chl, cmap='viridis', shading='auto', vmin=0, vmax=10)
        self.cs = ax.contour(xx, yy, self.chl, levels=[self.chl_ref])
        ax.plot(self.position[:,0], self.position[:,1], 'r', linewidth=3)
        ax.set_aspect('equal')
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        cp = fig.colorbar(p, cax=cax)
        cp.set_label("Chl a concentration [mm/mm3]")
        ax.set_xlabel("Longitude (degrees E)")
        ax.set_ylabel("Latitude (degrees N)")
        #plt.grid(True)

        return fig

################################################################################ PLOT GRADIENT #######################################################
    def gradient_comparison(self):

        # gt => ground truth
        self.gt_gradient = np.zeros([self.gradient.shape[0], 2])
        dot_prod_cos = np.zeros(self.gradient.shape[0])

        if self.gradient.shape[1] == 2:

            # Ground truth gradient
            gt_grad = np.gradient(self.chl)
            gt_grad_norm = np.sqrt(gt_grad[0]**2 + gt_grad[1]**2)
            everywhere_gradient = (RegularGridInterpolator((self.lon, self.lat), gt_grad[0]/gt_grad_norm),
                        RegularGridInterpolator((self.lon, self.lat), gt_grad[1]/gt_grad_norm))

            # Compute ground truth gradients
            for i in range(self.vector_length):
                x = self.meas_per*i

                self.gt_gradient[i, 0] = everywhere_gradient[0]((self.position[x,0], self.position[x,1]))
                self.gt_gradient[i, 1] = everywhere_gradient[1]((self.position[x,0], self.position[x,1]))
                dot_prod_cos[i] = np.dot(self.gradient[i], self.gt_gradient[i]) / (np.linalg.norm(self.gradient[i]) * np.linalg.norm(self.gt_gradient[i]))

            # Determine gradient angle
            gt_grad_angles = np.arctan2(self.gt_gradient[:, 1],self.gt_gradient[:, 0])
            pos = np.where(np.abs(np.diff(gt_grad_angles)) >= 4)[0]+1
            gt_grad_angles = np.insert(gt_grad_angles, pos, np.nan)
            it_gt_grad = np.insert(self.it, pos, np.nan)

            grad_angles = np.arctan2(self.gradient[:, 1],self.gradient[:, 0])
            pos = np.where(np.abs(np.diff(grad_angles)) >= 4)[0]+1
            grad_angles = np.insert(grad_angles, pos, np.nan)
            it_grad = np.insert(self.it, pos, np.nan)

        # Plot gradient angle
        fig, ax = plt.subplots(figsize=(15, 3))
        plt.plot(it_grad[self.zoom1_start:self.zoom1_end], grad_angles[self.zoom1_start:self.zoom1_end], 'k-', linewidth=1, label='GP Estimation')
        plt.plot(it_gt_grad[self.zoom1_start:self.zoom1_end], gt_grad_angles[self.zoom1_start:self.zoom1_end], 'r-', linewidth=1, label='Ground truth')
        plt.xlabel('Mission time [h]')
        plt.ylabel('Gradient [rad]')
        plt.legend(loc=4, shadow=True)
        plt.grid(True)
        plt.axis([self.it[self.zoom1_start], self.it[self.zoom1_end], -4, 3.4])

        return fig

################################################################################ PLOT CHL #######################################################
    def chl_comparison(self):
        
        fig, ax = plt.subplots(figsize=(15, 3))
        plt.plot(self.it[self.zoom1_start:self.zoom1_end], self.measurements[self.zoom1_start:self.zoom1_end], 'k-', linewidth=1, label="Measurements")
        plt.plot(self.it[self.zoom1_start:self.zoom1_end], np.tile(self.chl_ref, (self.zoom1_end-self.zoom1_start)), 'r-', label="Reference")
        if self.idx_trig > self.idx_start:
            plt.plot(np.tile(self.it[self.idx_trig], 10), np.linspace(np.min(self.measurements), self.chl_ref*1.4, 10), 'r--')
        plt.xlabel('Mission time [h]')
        plt.ylabel('Concentration\n [mm/mm3]')
        plt.axis([self.it[self.zoom1_start], self.it[self.zoom1_end], 6.5, 8]) # 6,9 and 2,9
        plt.legend(loc=4, shadow=True)
        plt.grid(True)

        return fig

################################################################################ PLOT DISTANCE #######################################################
    def distance_to_front(self):

        # Array to store distance
        dist = np.zeros(self.position.shape[0])

        chl_front = self.get_front_line()
        tree = spatial.KDTree(chl_front)
        projection = np.zeros(self.position.shape)

        for ind, point in enumerate(self.position):
            # Closest two points - v, w - between true path and ref path
            __ ,index = tree.query(point, 2)
            v = chl_front[index[0]]
            w = chl_front[index[1]]

            # Find projection of point p onto line, where t = [(p-v) . (w-v)] / |w-v|^2
            l2 = np.linalg.norm(v - w)**2  # |w-v|^2
            t = max(0, min(1, np.dot(point - v, w - v) / l2))
            projection = v + t * (w - v)

            # Compute distance with point to segment
            distance = geopy.distance.geodesic(point, projection).m
            dist[ind] = distance

        # Create special time vector that fits position
        time_position = np.linspace(self.start_time, self.end_time, self.position.shape[0])

        fig, ax = plt.subplots(figsize=(15, 3))
        plt.plot(time_position,dist,'k')
        plt.xlabel('Mission time [h]')
        plt.ylabel('Distance [m]')
        if self.idx_trig_time>self.idx_start:
            plt.plot(np.tile(self.idx_trig_time, 10), np.linspace(np.max(dist), 0, 10), 'r--')
        plt.grid(True)
        plt.axis([self.it[0], self.it[-1], 0, 300]) # 0,370 and 0,3100
        
        return fig

################################################################################# PLOT ZOOM map 1 with gradient
    def zoom1(self):
        # Square of space we want
        lat_start = 21.1
        lat_end = 21.17
        lon_start = 61.525
        lon_end = 61.56

        # Plot mesh with improved resolution using interpolation
        interpolating_function = RegularGridInterpolator((self.lon,self.lat), self.chl)
        fig, ax = plt.subplots(figsize=(15, 6))
        xx, yy = np.meshgrid(np.linspace(self.lon[0], self.lon[-1], 8*self.lon.shape[0]), np.linspace(self.lat[0], self.lat[-1], 8*self.lat.shape[0]))
        field = interpolating_function((xx, yy))
        
        # Plot contour and trajectory
        p = ax.pcolormesh(xx, yy, field, cmap='viridis', shading='auto', vmin=0, vmax=10)
        cs = ax.contour(xx, yy, field, levels=[self.chl_ref])
        ax.plot(self.position[self.zoom1_start:self.zoom1_end,0], self.position[self.zoom1_start:self.zoom1_end,1], 'r', linewidth=3, zorder = 1)

        ax.set_aspect('equal')
        ax.set_xlabel("Distance along longitude (km)")
        ax.set_ylabel("Distance along latitude (km)")
        ax.set_xlim([lat_start, lat_end])
        ax.set_ylim([lon_start, lon_end])
        #plt.grid(True)

        # Plot gradient arrows
        for index in range(self.measurements.shape[0]):
            if index % 550 == 0 :
                x=self.position[self.meas_per*index,0]
                y=self.position[self.meas_per*index,1]
                dx=0.003*self.gradient[index][0] / np.linalg.norm(self.gradient[index])
                dy=0.003*self.gradient[index][1] / np.linalg.norm(self.gradient[index])
                dx_gt = 0.003*self.gt_gradient[index][0] / np.linalg.norm(self.gt_gradient[index])
                dy_gt = 0.003*self.gt_gradient[index][1] / np.linalg.norm(self.gt_gradient[index])
                arrow1 = ax.arrow(x, y, dx, dy, width=.00008, color='black', zorder = 2)
                arrow2 = ax.arrow(x, y, dx_gt, dy_gt, width=.00008, color='red', zorder = 2)

        # Legend for the arrows
        h,l = plt.gca().get_legend_handles_labels()
        h.append(arrow1)
        h.append(arrow2)
        l.append('Estimated gradient')
        l.append('Ground truth')
        plt.legend(h,l, handler_map={mpatches.FancyArrow : HandlerArrow()})
        
        # Change distance from degres to meters
        ax.set_xticks([21.1, 21.11, 21.12, 21.13, 21.14, 21.15, 21.16, 21.17])
        ax.set_xticklabels([0, 0.5, 1.1, 1.6, 2.1, 2.7, 3.2, 3.7]) #([0, 530, 1060 , 1590, 2120, 2650, 3180, 3710])
        ax.set_yticks([61.525, 61.53, 61.535, 61.54, 61.545, 61.55, 61.555, 61.56])
        ax.set_yticklabels([0, 0.5, 1.1, 1.7, 2.2, 2.8, 3.3, 3.9]) #([0, 556, 1112, 1668, 2224, 2780, 3336, 3892, 4448])
        
        return fig
        
        
################################################################################ PLOT Control Law #######################################################
    def control_input(self):
        
        # Determine control seek and follow
        alpha_seek = 50
        alpha_follow = 1
        chl_error = (self.chl_ref* np.ones(self.measurements.shape[0]) - self.measurements)
        self.control_follow = alpha_follow * np.ones(self.measurements.shape[0])
        self.control_seek = alpha_seek * chl_error 

        for index in range(self.control_follow.shape[0]):
            total_control = [self.control_follow[index], self.control_seek[index]]
            self.control_follow[index] = self.control_follow[index] / np.linalg.norm(total_control)
            self.control_seek[index] = self.control_seek[index] / np.linalg.norm(total_control)

        # Create special time vector that fits position
        time_control = np.linspace(self.start_time, self.end_time, self.measurements.shape[0])

        # Plot control angle
        fig, ax = plt.subplots(figsize=(15, 3))
        plt.plot(time_control[self.zoom2_start:self.zoom2_end], np.abs(self.control_seek[self.zoom2_start:self.zoom2_end]), 'b-', linewidth=1, label='Control seek')
        plt.plot(time_control[self.zoom2_start:self.zoom2_end], self.control_follow[self.zoom2_start:self.zoom2_end], 'k-', linewidth=1, label='Control follow')
        plt.xlabel('Mission time [h]')
        plt.ylabel('Control law')
        plt.legend(loc=4, shadow=True)
        plt.grid(True)
        plt.axis([self.it[self.zoom2_start], self.it[self.zoom2_end], -2, 1.5])

        return fig
        
################################################################################# PLOT ZOOM map 2 with control
    def zoom2(self):
        # Square of space we want
        lon_start = 21.142
        lon_end = 21.152
        lat_start = 61.534
        lat_end = 61.539

        # Get perpendicular vector to the gradient: R_{pi/2} * gradient
        perpendicular_gradient = np.zeros([self.gradient.shape[0], 2])
        rotation = np.array([[0, -1],[1, 0]])
        perpendicular_gradient = rotation.dot(self.gradient.T).T


        # Plot mesh with improved resolution using interpolation
        interpolating_function = RegularGridInterpolator((self.lon,self.lat), self.chl)
        fig, ax = plt.subplots(figsize=(15, 6))
        xx, yy = np.meshgrid(np.linspace(self.lon[0], self.lon[-1], 8*self.lon.shape[0]), np.linspace(self.lat[0], self.lat[-1], 8*self.lat.shape[0]))
        field = interpolating_function((xx, yy))
        
        # Plot contour and trajectory
        p = ax.pcolormesh(xx, yy, field, cmap='viridis', shading='auto', vmin=0, vmax=10)
        cs = ax.contour(xx, yy, field, levels=[self.chl_ref], zorder = 1)
        ax.plot(self.position[self.zoom2_start:self.zoom2_end,0], self.position[self.zoom2_start:self.zoom2_end,1], 'r', linewidth=2, zorder = 2)
        ax.set_aspect('equal')
        ax.set_xlabel("Distance along longitude (m)")
        ax.set_ylabel("Distance along latitude (m)")
        ax.set_xlim([lon_start, lon_end])
        ax.set_ylim([lat_start, lat_end])
        #plt.grid(True)

        # Plot control arrows
        for index in range(self.measurements.shape[0]):
            if index % 80 == 0 :
                x = self.position[self.meas_per*index,0]
                y = self.position[self.meas_per*index,1]

                dx_seek = 0.0004*self.control_seek[index]*self.gradient[index][0] / np.linalg.norm(self.gradient[index])
                dy_seek = 0.0004*self.control_seek[index]*self.gradient[index][1] / np.linalg.norm(self.gradient[index])

                dx_follow = 0.0004*self.control_follow[index]*perpendicular_gradient[index][0] / np.linalg.norm(self.gradient[index]) 
                dy_follow = 0.0004*self.control_follow[index]*perpendicular_gradient[index][1] / np.linalg.norm(self.gradient[index])

                arrow1 = ax.arrow(x, y, dx_seek, dy_seek, width=.00002, color='blue', zorder = 3)
                arrow2 = ax.arrow(x, y, dx_follow, dy_follow, width=.00002, color='black', zorder = 3)

        
        # Legend for the arrows
        h,l = plt.gca().get_legend_handles_labels()
        h.append(arrow1)
        h.append(arrow2)
        l.append('Control seek')
        l.append('Control follow')
        plt.legend(h,l, handler_map={mpatches.FancyArrow : HandlerArrow()})

        # Change distance from degres to meters
        ax.set_xticks([21.142, 21.144, 21.146, 21.148, 21.150, 21.152])
        ax.set_xticklabels([0, 105, 211, 317, 423, 529])
        ax.set_yticks([61.534, 61.536, 61.538])
        ax.set_yticklabels([0, 111, 223])
        
        return fig