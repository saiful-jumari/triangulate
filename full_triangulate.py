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

fx, fy = 250, 250
k = camera_intrinsics(fx, fy)

# transform 3d point in global frame to local frame of camera with pose (x, y, theta)
# defined in global frame
def global_to_local(x, y, theta):
    s = np.sin(theta)
    c = np.cos(theta)
    mat = [[c,  -s, 0,  x],
           [s,  c,  0,  y],
           [ 0,  0, 1,  0],
           [ 0,  0, 0,  1]]
    return linalg.inv(np.matrix(mat))

# generate random 3d point in 1st camera frame (i.e. global frame)
def random_3d_point():
    x = random.uniform(0, 7)
    y = random.uniform(-3, 3)
    z = random.uniform(0.1, 5)
    return np.matrix([x, y, z, 1]).T

# given 3d point, simulate actual appearance on camera
def project_point_to_camera(point, x, y, theta, k, noise=0):
    # transform to local camera coords
    point_local_cam = world_to_cam * global_to_local(x, y, theta) * point

    # divide by depth
    x = point_local_cam.item(0, 0)
    y = point_local_cam.item(1, 0)
    z = point_local_cam.item(2, 0)
    point_local_cam_descale = np.matrix([x/z, y/z, 1, 1]).T

    # noise
    epsilon = np.matrix([np.random.normal(scale=noise), np.random.normal(scale=noise), 0]).T

    # transform to pixel coordinates, add noise
    px = k * point_local_cam_descale + epsilon
    return [px.item(0, 0), px.item(1, 0)]

def generate_random_image_measurements(features, x, y, theta):
    measurements = []
    for f in features:
        m = project_point_to_camera(f, x, y, theta)
        measurements.append(m)
    return measurements

def get_fundamental_matrix(x, y, theta, k_3d):
    K = k_3d * world_to_cam_3
    K_inv = linalg.inv(K)

    phi = np.arctan2(y, x)
    gamma = theta - phi
    cp, sp = np.cos(phi), np.sin(phi)
    cg, sg = np.cos(gamma), np.sin(gamma)

    # given x0, x unit vectors to feature at origin and non-origin:
    # x0^T * E * x = 0
    # Using K * x = u,
    # => x = K^-1 * u
    # thus,
    # u0^T * [(K^-1)^T * E * (K^-1)] * u = 0
    E = [[0, 0,  sp],
         [0, 0, -cp],
         [sg, cg, 0]]
    return K_inv.T * np.matrix(E) * K_inv

def solve_w_opencv(pts0, pts1, x, y, theta, k):
    k_3d = np.delete(k, 3, 1)

    # satisfies u0^T * F * u
    F = get_fundamental_matrix(x, y, theta, k_3d)

    u0 = pts0[0] + [1]
    u = pts1[0] + [1]
    print("Check fundamental constraint: ", np.matrix(u0) * F * np.matrix(u).T)

    n = len(pts0)
    pts0_format = np.reshape(pts0, (1, n, 2))
    pts1_format = np.reshape(pts1, (1, n, 2))
    corrected_pts1, corrected_pts0  = cv2.correctMatches(F, pts1_format, pts0_format)

    # robot original position
    P0 = k * world_to_cam * global_to_local(0, 0, 0)

    # robot moved
    P = k * world_to_cam * global_to_local(x, y, theta)

    points_homo = cv2.triangulatePoints(np.array(P0), np.array(P), corrected_pts0, corrected_pts1)

    # convert homogeneous 3D to normal 3D coordinates
    points = []
    for p in points_homo.T:
        w = p[3]
        points.append(np.matrix([p[0]/w, p[1]/w, p[2]/w, 1]).T)
    return points

# percentage error
def error(feat, est):
    return linalg.norm(feat - est) * 100 / linalg.norm(feat)

def main():
    # num features to project onto each image (one at origin, one with relative pose)
    num_features = 500

    # pixel measurement noise
    noise = 8

    # relative pose of camera in 2nd image
    x = 0
    y = 2
    theta = 0.0

    # generate random features in 3D homogeneous coordinates
    features = []
    for i in range(num_features):
        features.append(random_3d_point())

    # simulate how features will appear on camera at both positions
    measurements0 = []
    measurements1 = []
    for f in features:
        measurements0.append(project_point_to_camera(f, 0, 0, 0, k, noise))
        measurements1.append(project_point_to_camera(f, x, y, theta, k, noise))

    # estimate 3d position of features from measurements and relative pose
    est_points = solve_w_opencv(measurements0, measurements1, x, y, theta, k)

    errs = []
    for i in range(num_features):
        errs.append(error(features[i], est_points[i]))
    print("Mean error: ", np.mean(errs))

main()
    