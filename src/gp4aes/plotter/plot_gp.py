import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the HDF5 file containing the processed data")

    return parser.parse_args()


def main(args):
    with h5.File(args.path, "r") as f:
        chl = f["chl"][()]
        lat = f["lat"][()]
        lon = f["lon"][()]
        X = f["X"][()]
        y = f["y"][()]
        x = f["x"][()]
        y_pred = f["y_pred"][()]
        std_pred = f["std_pred"][()]
        rel_error = f["rel_error"][()]
        av_rel_error = f.attrs["av_rel_error"]
        t_idx = f.attrs["t_idx"]

    # Prepare plots
    gt_grid = np.meshgrid(lon, lat, indexing='ij')
    pred_grid = np.meshgrid(x[0], x[1])

    y_pred = y_pred.reshape(pred_grid[0].shape)
    std_pred = std_pred.reshape(pred_grid[0].shape)
    rel_error = rel_error.reshape(pred_grid[0].shape)

    # Ground Truth
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    p_gt = plt.pcolormesh(gt_grid[0], gt_grid[1], chl[:,:,t_idx], cmap='viridis', shading='auto', vmin=0, vmax=10)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cp = fig.colorbar(p_gt, cax=cax)
    cp.set_label("Chl a density [mm/mm3]")
    ax.set_title("Ground Truth")
    ax.set_xlabel("Longitude (ºE)")
    ax.set_ylabel("Latitude (ºN)")

    # Mean Prediction - Clean
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    p_pred = plt.pcolormesh(pred_grid[0], pred_grid[1], y_pred, cmap='viridis', shading='auto', vmin=0, vmax=10)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cp = fig.colorbar(p_pred, cax=cax)
    cp.set_label("Chl a density [mm/mm3]")
    ax.set_xlabel("Longitude (ºE)")
    ax.set_ylabel("Latitude (ºN)")

    # Mean Prediction 2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    p_pred = plt.pcolormesh(pred_grid[0], pred_grid[1], y_pred, cmap='viridis', shading='auto', vmin=0, vmax=10)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cp = fig.colorbar(p_pred, cax=cax)
    cp.set_label("Chl a density [mm/mm3]")
    plt.scatter(X[:, 0], X[:, 1], y, linewidth=0.75, color='k')
    ax.set_title("Predicted Mean ")
    ax.set_xlabel("Longitude (ºE)")
    ax.set_ylabel("Latitude (ºN)")

    # Std Prediction
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    p_std = plt.pcolormesh(pred_grid[0], pred_grid[1], std_pred, cmap='viridis', shading='auto')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cp = fig.colorbar(p_std, cax=cax)
    cp.set_label("Chl a density [mm/mm3]")
    plt.scatter(X[:, 0], X[:, 1], y, linewidth=0.75, color='k')
    ax.set_title("Predicted Std [mm/mm3]")
    ax.set_xlabel("Longitude (ºE)")
    ax.set_ylabel("Latitude (ºN)")

    # Error Prediction
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    p_error = plt.pcolormesh(pred_grid[0], pred_grid[1], np.abs(rel_error), cmap='viridis', shading='auto', vmin=0, vmax=8)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cp = fig.colorbar(p_error, cax=cax)
    cp.set_label("Error [%]")
    plt.scatter(X[:, 0], X[:, 1], y, linewidth=0.75, color='k')
    ax.set_title("Relative error[%%]\nAverage error = %.3f" % av_rel_error)
    ax.set_xlabel("Longitude (ºE)")
    ax.set_ylabel("Latitude (ºN)")

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)