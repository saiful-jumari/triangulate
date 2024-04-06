import numpy as np
from numpy import linalg
import random
import cv2

world_to_cam = np.matrix([[0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [1, 0,  0, 0],
                          [0, 0,  0, 1]])

world_to_cam_3 = np.matrix([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0,  0]])

def camera_intrinsics(fx=1, fy=1, cx=0, cy=0):
    mat = [[fx, 0, cx, 0],
           [0, fy, cy, 0],
           [0,  0,  1, 0]]
    return np.matrix(mat)

def global_to_local(x, y, theta):
    s = np.sin(theta)
    c = np.cos(theta)
    mat = [[c,  -s, 0,  x],
           [s,  c,  0,  y],
           [ 0,  0, 1,  0],
           [ 0,  0, 0,  1]]
    return linalg.inv(np.matrix(mat))

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
    epsilon = np.matrix([np.random.normal(scale=noise), np.random.normal(scale=noise), 0]).T

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

def solve_w_opencv(pt0, pt, x, y, theta, fx, fy, cx, cy):
    k = camera_intrinsics(fx, fy, cx, cy)

    # robot unmoved
    P0 = k * world_to_cam * global_to_local(0, 0, 0)

    # robot moved
    P = k * world_to_cam * global_to_local(x, y, theta)

    x0 = np.array([pt0.item(0, 0), pt0.item(1, 0)])
    x = np.array([pt.item(0, 0), pt.item(1, 0)])
    point_homo = cv2.triangulatePoints(np.array(P0), np.array(P), x0, x)

    x = point_homo.item(0, 0)
    y = point_homo.item(1, 0)
    z = point_homo.item(2, 0)
    w = point_homo.item(3, 0)

    point = [x/w, y/w, z/w, 1]
    return np.matrix(point).T

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def solve_rand_point_exact(use_eigen=False):
    pt = random_3d_point()
    print("Rand point: ", pt.T)

    x, y, theta = 0.03, 0, 0.09
    fx, fy, cx, cy = 50, 50, 0, 0
    noise = 5
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

def solve_rand_point_exact_open_cv():
    pt = random_3d_point()
    print("Rand point: ", pt.T)

    x, y, theta = 0.03, 0, 0.3
    fx, fy, cx, cy = 50, 50, 0, 0
    noise = 0
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

    est_3d_point = solve_w_opencv(px0, px, x, y, theta, fx, fy, cx, cy)

    print("Estimated point: ", est_3d_point.T)

    err = dist(pt, est_3d_point)
    percent_err = err * 100 / np.linalg.norm(pt)
    print("% Error: ", percent_err)
    return percent_err

def compute_local_image_frame_transform(epipole, u):
    e1 = epipole.item(0, 0)
    e2 = epipole.item(1, 0)
    e3 = epipole.item(2, 0)

    u1 = u.item(0, 0)
    u2 = u.item(1, 0)

    L = np.matrix([[1, 0, -u1],
                   [0, 1, -u2],
                   [0, 0,   1]])

    # solve a * sin (theta) + b * cos(theta) = 0
    # -> use R * cos (alpha) * sin (theta) + R * sin (alpha) * cos (theta) = 0 (R-formula)
    # R sin (theta + alpha) = 0
    # -> tan (alpha) = b / a
    a = e1 - e3 * u1
    b = e2 - e3 * u2

    alpha = np.arctan2(b, a)
    theta = 3.1415 - alpha

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.matrix([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]])
    return R * L

# compute fundamental matrix from relative pose and camera matrix
def get_fundamental_matrix(x, y, theta, fx, fy, cx, cy):
    k = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

    K = np.matrix(k) * world_to_cam_3
    K_inv = linalg.inv(K)

    phi = np.arctan2(y, x)
    gamma = theta - phi
    cp, sp = np.cos(phi), np.sin(phi)
    cg, sg = np.cos(gamma), np.sin(gamma)

    # given x0, x unit vectors to feature at origin and non-origin:
    # x0^T * E * x = 0
    # K * x = u => x = K^-1 * u
    E = [[0, 0, sp],
         [0, 0, -cp],
         [sg, cg, 0]]

    return K_inv.T * np.matrix(E).T * K_inv

# perform harley and sturm technique to identify best estimate of noisy observations in two images
# that line on an epipolar plane defined by the relative pose between the two camera
def find_measurements_on_epipolar_plane_poly(px0, px, x, y, theta, fx, fy, cx, cy):
    k = camera_intrinsics(fx, fy, cx, cy)

    # project the 2d position of the second camera onto the local frame of the first
    epipole0 = k * world_to_cam * np.matrix([x, y, 0, 1]).T
    T0 = compute_local_image_frame_transform(epipole0, px0)

    # project the 2d position of the first camera onto the local frame of the second
    c = np.cos(theta)
    s = np.sin(theta)
    epipole = k * world_to_cam * global_to_local(x, y, theta) * np.matrix([0, 0, 0, 1]).T
    T = compute_local_image_frame_transform(epipole, px)

    # sanity check identities
    print("Check local image coord trans should be (0, 0, 1):", (T0 * px0).T)
    print("Check local image coord trans should be (0, 0, 1):", (T * px).T)

    F = get_fundamental_matrix(x, y, theta, fx, fy, cx, cy)

    # sanity check essential matrix 0 for noiseless cass
    print("Check fundamental identity is 0: ", px.T * F * px0)

    transformed_F = T * F * linalg.inv(T0)

    transformed_epipole0 = T0 * epipole0
    transformed_epipole = T * epipole

    transformed_epipole0 = transformed_epipole0 / transformed_epipole0.item(0, 0)
    transformed_epipole = transformed_epipole / transformed_epipole.item(0, 0)
    print("Check transformed epipole0 (1, 0, f): ", transformed_epipole0.T)
    print("Check transformed epipole (1, 0, f'): ", transformed_epipole.T)

    f0 = transformed_epipole0.item(2, 0)
    f = transformed_epipole.item(2, 0)

    a = transformed_F.item(1, 1)
    b = transformed_F.item(1, 2)
    c = transformed_F.item(2, 1)
    d = transformed_F.item(2, 2)

    # sanity check transformed F and abcd
    F_check = [[f0*f*d, -f*c, -f*d],
               [-f0*b, a, b],
               [-f0*d, c, d]]

    # transformed F and abcd F seems off for the 1st row??
    print("Transformed F:", transformed_F, "\n abcd F:", np.matrix(F_check))

def solve_via_poly():
    pt = random_3d_point()
    print("Rand point: ", pt.T)

    x, y, theta = 0.1, 0.1, 0
    fx, fy, cx, cy = 50, 50, 0, 0
    noise = 0
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

    find_measurements_on_epipolar_plane_poly(px0, px, x, y, theta, fx, fy, cx, cy)

def main():
    attempts = 1

    eig_errs = []
    no_eig_errs = []
    for i in range(attempts):
        solve_via_poly()
        # err_no_eig = solve_rand_point_exact_open_cv()
        # no_eig_errs.append(err_no_eig)
    
    # mean_no_eig = np.mean(no_eig_errs)
    # print("No Eig % error: ", mean_no_eig)

main()
