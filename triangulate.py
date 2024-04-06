import numpy as np
from numpy import linalg
import random

world_to_cam = np.matrix([[0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [1, 0,  0, 0],
                          [0, 0,  0, 1]])


def camera_intrinsics(fx=1, fy=1, cx=0, cy=0):
    mat = [[fx, 0, cx, 0],
           [0, fy, cy, 0],
           [0,  0,  1, 0]]
    return np.matrix(mat)


def global_to_local(x, y, theta):
    s = np.sin(theta)
    c = np.cos(theta)
    mat = [[ c,  s, 0, -x],
           [-s,  c, 0, -y],
           [ 0,  0, 1,  0],
           [ 0,  0, 0,  1]]
    return np.matrix(mat)


def random_3d_point():
    x = random.uniform(0, 7)
    y = random.uniform(-3, 3)
    z = random.uniform(0.1, 5)
    return np.matrix([x, y, z, 1]).T


# given 3d point, where will it appear on the image?
def project_point_to_cam(point, x, y, theta, fx=1, fy=1, cx=0, cy=0, noise=0):
    # transform to local camera coords
    point_local_cam = world_to_cam * global_to_local(x, y, theta) * point

    # divide by z
    x = point_local_cam.item(0, 0)
    y = point_local_cam.item(1, 0)
    z = point_local_cam.item(2, 0)
    point_local_cam_descale = np.matrix([x/z, y/z, 1, 1]).T

    # convert to pixel coords
    k = camera_intrinsics(fx, fy, cx, cy)

    # add noise
    epsilon = np.matrix([np.random.normal(scale=noise), np.random.normal(scale=noise), 1]).T

    return k * point_local_cam_descale + epsilon

def triangulate(pt0, pt, x, y, theta, fx, fy, cx, cy, use_eigen=False):
    k = camera_intrinsics(fx, fy, cx, cy)

    # robot unmoved
    P0 = k * world_to_cam * global_to_local(0, 0, 0)

    # robot moved
    P = k * world_to_cam * global_to_local(x, y, theta)

    P0_1 = np.array([P0.item(0, 0), P0.item(0, 1), P0.item(0, 2), P0.item(0, 3)])
    P0_2 = np.array([P0.item(1, 0), P0.item(1, 1), P0.item(1, 2), P0.item(1, 3)])
    P0_3 = np.array([P0.item(2, 0), P0.item(2, 1), P0.item(2, 2), P0.item(2, 3)])
    u0, v0 = pt0.item(0, 0), pt0.item(1, 0)

    P_1 = np.array([P.item(0, 0), P.item(0, 1), P.item(0, 2), P.item(0, 3)])
    P_2 = np.array([P.item(1, 0), P.item(1, 1), P.item(1, 2), P.item(1, 3)])
    P_3 = np.array([P.item(2, 0), P.item(2, 1), P.item(2, 2), P.item(2, 3)])
    u, v = pt.item(0, 0), pt.item(1, 0)

    r1 = v0 * P0_3 - P0_2
    r2 = P0_1 - u0 * P0_3
    r3 = v * P_3 - P_2
    r4 = P_1 - u * P_3

    mat = np.matrix([r1, r2, r3, r4])

    if not use_eigen:
        U, S, Vt = np.linalg.svd(mat)
        soln = Vt[3]
        w = soln.item(0, 3)
        return (soln / w).T
    else:
        eigvals, eigvecs = np.linalg.eig(mat.T * mat)
        
        smallest_eig = 10000
        idx = -1
        for i in range(len(eigvals)):
            if abs(eigvals[i]) < smallest_eig:
                idx = i
                smallest_eig = eigvals[i]
        
        soln = eigvecs[i]
        w = soln.item(0, 3)
        return (soln / w).T

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def solve_rand_point_exact(use_eigen=False):
    pt = random_3d_point()
    print("Rand point: ", pt.T)

    x, y, theta = 0.03, 0, 0.09
    fx, fy, cx, cy = 50, 50, 0, 0
    noise = 0.00
    print("Pixel noise: ", noise)
    px0 = project_point_to_cam(pt, 0, 0, 0, fx, fy, cx, cy, noise)
    px = project_point_to_cam(pt, x, y, theta, fx, fy, cx, cy, noise)

    print("Observed loc 1: ", px0.T)
    print("Observed loc 2:", px.T)

    # assume 0.5cm trans noise
    x_noise = np.random.normal(0.01) * 0
    y_noise = np.random.normal(0.01) * 0

    # assume 0.5 deg angular noise
    theta_noise = np.random.normal(0.009) * 0

    est_3d_point = triangulate(px0, px, x + x_noise, y + y_noise, theta + theta_noise, fx, fy, cx, cy, use_eigen)
    print("Estimated point: ", est_3d_point.T)

    err = dist(pt, est_3d_point)
    percent_err = err * 100 / np.linalg.norm(pt)
    print("% Error: ", percent_err)
    return percent_err

def main():
    attempts = 1000

    eig_errs = []
    no_eig_errs = []
    for i in range(attempts):
        err_no_eig = solve_rand_point_exact(False)
        no_eig_errs.append(err_no_eig)
    
    mean_no_eig = np.mean(no_eig_errs)
    print("No Eig % error: ", mean_no_eig)

main()
