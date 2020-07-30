import os
import json
import numpy as np
import cv2


def calibrate_stereo(img_folder, img_idx, write_folder, \
                     chessboard_size=(9, 6), square_size=0.02):
    cb_row, cb_col = chessboard_size
    num_camera = 2
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cb_col * cb_row,3), np.float32)
    objp[:,:2] = square_size * np.mgrid[0:cb_row,0:cb_col].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_cam0 = [] # 2d points in image plane.
    imgpoints_cam1 = []

    images = ["cam0_%06d.jpg" % i for i in img_idx]

    for img_i in img_idx:
        gray_cam = []
        ret_cam = []
        corners_cam = []
        for camera_idx in range(num_camera):
            fname = "cam%d_%06d.jpg" % (camera_idx, img_i)
            img_path = os.path.join(img_folder, fname)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            gray_cam.append(gray)
            ret_cam.append(ret)
            corners_cam.append(corners)
        if (ret_cam[0] == True) and (ret_cam[1] == True):
            objpoints.append(objp)
            corners_refined_cam0 = cv2.cornerSubPix(gray_cam[0], corners_cam[0], (11,11), (-1,-1), criteria)
            corners_refined_cam1 = cv2.cornerSubPix(gray_cam[1], corners_cam[1], (11,11), (-1,-1), criteria)
            imgpoints_cam0.append(corners_refined_cam0)
            imgpoints_cam1.append(corners_refined_cam1)

    # calibrate camera intrinsics and distortions      
    _, mtx0, dist0, _, _ = cv2.calibrateCamera(objpoints, imgpoints_cam0, gray.shape[::-1], None, None)
    _, mtx1, dist1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_cam1, gray.shape[::-1], None, None)

    # calibrate relative pose (1st camera relative to 2nd camera)
    ret, mtx0, dist0, mtx1, dist1, R, T, _, _ = cv2.stereoCalibrate(objpoints, imgpoints_cam0, imgpoints_cam1, \
                                                                    mtx0, dist0, mtx1, dist1, \
                                                                    gray.shape[::-1])

    cam0 = {}
    cam0["int"] = mtx0.tolist()
    cam0["ext"] = np.hstack((R.T, -R.T @ T)).tolist()
    cam0["dist"] = dist0.tolist()
    with open(os.path.join(write_folder, "cam0.json"), 'w') as json_file:
        json.dump(cam0, json_file)

    cam1 = {}
    cam1["int"] = mtx1.tolist()
    cam1["ext"] = np.eye(4)[0:3, :].tolist()
    cam1["dist"] = dist1.tolist()
    with open(os.path.join(write_folder, "cam1.json"), 'w') as json_file:
        json.dump(cam1, json_file)
    

if __name__ == "__main__":
    
    img_folder = os.path.join("data", "real", "checkerboard")
    img_idx = [71, 90, 109, 129, 150, 168, 205, 260, 306, 360, 388, 455, 547]
    write_folder = os.path.join("data", "real", "single_human")

    calibrate_stereo(img_folder, img_idx, write_folder, \
                     chessboard_size=(9, 6), square_size=0.02)
