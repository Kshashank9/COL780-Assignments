import os
import sys
import cv2
import numpy as np
import math
import random

# Get the directory path from command line arguments
dir_path = sys.argv[1]

# Name the output panorama file
name = sys.argv[2]

# Loop through all the files in the directory
images = []
gr_imgs = []

path = []
for filename in os.listdir(dir_path):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read the image using OpenCV
        img_path = os.path.join(dir_path, filename)
        path.append(img_path)
        
path = sorted(path, key=lambda x: int(x.split()[-1][:-4]))


for i in range(len(path)):
    img = cv2.imread(path[i])
    gr_img = cv2.imread(path[i], cv2.IMREAD_GRAYSCALE)
    images.append(img)
    gr_imgs.append(gr_img)


def get_corners(img):
    
    # sigma = 3
    gaussian = np.array([[0.01134374, 0.08381951, 0.01134374],
                [0.08381951, 0.61934703, 0.08381951],
                [0.01134374, 0.08381951, 0.01134374]])
    img = cv2.filter2D(img, -1, gaussian)

    # Compute the derivatives
    dx2 = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=3)
    dy2 = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=3)
    dxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

    # Apply Gaussian filter to the products of derivatives
    ksize = 3  # Kernel size for the Gaussian filter
    sigma = 1  # Standard deviation for the Gaussian filter
    dx2 = cv2.GaussianBlur(dx2, (ksize, ksize), sigma)
    dy2 = cv2.GaussianBlur(dy2, (ksize, ksize), sigma)
    dxy = cv2.GaussianBlur(dxy, (ksize, ksize), sigma)

    s0, s1 = img.shape
    
    corner_response = np.zeros(img.shape)
    
    k = 0.15
    for i in range(1, s0-1, 2):
        for j in range(1, s1-1, 2):
            win_dx2 = dx2[i-2:i+3, j-2:j+3]
            win_dy2 = dy2[i-2:i+3, j-2:j+3]
            win_dxy = dxy[i-2:i+3, j-2:j+3]
            s_x2 = np.sum(win_dx2)
            s_y2 = np.sum(win_dy2)
            s_xy = np.sum(win_dxy)
            det = (s_x2 * s_y2) - (s_xy * s_xy)
            tr = s_x2 + s_y2
            R = det - (k * tr * tr)
            corner_response[i][j] = R
    

    # Threshold the corner response image
    threshold = 0.12 * corner_response.max()
    corners = np.zeros_like(corner_response)
    corners[corner_response > threshold] = 255
    # corners[corner_response < threshold] = 0
    return corners

# Finding the difference between two patches
def ssd(patch1, patch2):
    return np.sum((patch1 - patch2) ** 2)

def ransac_affine(pts1, pts2):
    mean1 = np.mean(pts1, axis=0)
    mean2 = np.mean(pts2, axis=0)

    # Compute the centered keypoints
    pts1_centered = pts1 - mean1
    pts2_centered = pts2 - mean2

    # Compute the cross-covariance matrix
    cross_covariance = np.dot(pts1_centered.reshape(-1, 2).T, pts2_centered.reshape(-1, 2))

    # Compute the singular value decomposition of the cross-covariance matrix
    U, s, Vt = np.linalg.svd(cross_covariance)

    # Compute the rotation matrix and translation vector
    R = np.dot(Vt.T, U.T)
    t = mean1.T - np.dot(R, mean2.T)

    # Combine the rotation matrix and translation vector into a 3x3 affine matrix
    affine_matrix = np.zeros((3, 3))
    affine_matrix[:2, :2] = R
    affine_matrix[:2, 2] = t.reshape(-1)
    affine_matrix[2, 2] = 1.0

    # Extract the 2x3 affine transformation matrix from the 3x3 affine matrix
    affine_transform = affine_matrix[:2, :]
    
    return affine_transform

def best_affine(pts1, pts2, num_iterations, threshold):
    num_points = len(pts1)
    inlier_mask = np.zeros(num_points, dtype=bool)
    best_inlier_mask = np.zeros(num_points, dtype=bool)
    max_num_inliers = 0
    max_M = None

    for i in range(num_iterations):
        # Select a random sample of 3 matching points
        indices = np.random.choice(num_points, 3, replace=False)
        src_pts = np.float32([pts1[j] for j in indices])
        dst_pts = np.float32([pts2[j] for j in indices])

        # Compute the affine transformation matrix
        M = ransac_affine(src_pts, dst_pts)

        # Apply the affine transformation to all matching points in the second image
        transformed_pts = cv2.transform(np.float32([pts2]), M)

        # Calculate the Euclidean distances between the transformed points and their matches in the first image
        distances = np.sqrt(np.sum((transformed_pts - pts1)**2, axis=2)).flatten()

        # Mark points as inliers if their distance is below the threshold
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        # Update the inlier mask if the current transformation has more inliers than the previous best
        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            best_inlier_mask = inliers
            max_M = M

    # Used all the best inlier points to compute the affine transformation matrix
    src_pts = np.float32([pts1[j] for j in range(num_points) if best_inlier_mask[j]])
    dst_pts = np.float32([pts2[j] for j in range(num_points) if best_inlier_mask[j]])
    max_M = ransac_affine(src_pts, dst_pts)

    return max_M

thresh = 500
patch_size = (5, 5, 3)

img1 = None
img2 = None

im1 = None
im2 = None

for q in range(1, len(gr_imgs)):
    if q == 1:
        img1 = gr_imgs[0]
        img2 = gr_imgs[1]
        im1 = images[0]
        im2 = images[1]

    else:
        img2 = gr_imgs[q]
        im2 = images[q]

    corners1 = get_corners(img1)
    corners2 = get_corners(img2)

    corner1_pts = []
    for i in range(len(corners1)):
        for j in range(len(corners1[0])):
            if corners1[i][j] > 0:
                corner1_pts.append((j, i))

    corner2_pts = []
    for i in range(len(corners2)):
        for j in range(len(corners2[0])):
            if corners2[i][j] > 0:
                corner2_pts.append((j, i))

    corners1 = [[], []]
    for i in range(len(corner1_pts)):
        corners1[0].append(corner1_pts[i][0])
        corners1[1].append(corner1_pts[i][1])

    corners2 = [[], []]
    for i in range(len(corner2_pts)):
        corners2[0].append(corner2_pts[i][0])
        corners2[1].append(corner2_pts[i][1])

    corners1 = np.array(corners1)
    corners2 = np.array(corners2)

    matches = []

    for i in range(len(corner1_pts)):
        x, y = corner1_pts[i][0], corner1_pts[i][1]
        patch1 = im1[y-2:y+3, x-2:x+3]
        best_match = None
        best_ssd = np.inf
    
        # Hyperparameters that can be tuned according to the scenario
        y_r = y-350
        y_r2 = y+350
    
        x_r = x-350
        x_r2 = x+300
    
        for j in range(len(corner2_pts)):
            x2, y2 = corner2_pts[j][0], corner2_pts[j][1]
            if y2>=y_r and y2<=y_r2 and x2>=x_r and x2<=x_r2:
                patch2 = im2[y2-2:y2+3, x2-2:x2+3]
                if patch1.shape == patch_size and patch2.shape == patch_size:
                    ssd_val = ssd(patch1, patch2)
                    if ssd_val < best_ssd:
                        best_match = (x2, y2)
                        best_ssd = ssd_val
        if best_match is not None and best_ssd<thresh:
            matches.append((x, y, best_match[0], best_match[1], best_ssd))


    pts1 = []
    pts2 = []

    for i in range(len(matches)):
        pts1.append((matches[i][0], matches[i][1]))
        pts2.append((matches[i][2], matches[i][3]))
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    affine_transform = best_affine(pts1, pts2, 15000, 5.0)

    panorama = None
    
    print(affine_transform)

    if round(affine_transform[0][2]) < 0 or round(affine_transform[1][2]) < 0:
    
        affine_transform[0][2] = -1 * affine_transform[0][2]
        affine_transform[1][2] = -1 * affine_transform[1][2]
        
        # Warp the second image using the computed t@ransformation
        panorama = cv2.warpAffine(im1, affine_transform, (im1.shape[1] + round(affine_transform[0][2]), im1.shape[0] + round(affine_transform[1][2])))

        # Place the first image onto the panorama
        panorama[:im2.shape[0], :im2.shape[1]] = im2

        #Proposed Change
        #panorama = cv2.warpAffine(im1, affine_transform, (im1.shape[1] + round(affine_transform[0][2])), (im1.shape[0] + round(affine_transform[1][2])))

    
    else:
        # Warp the second image using the computed transformation
        panorama = cv2.warpAffine(im2, affine_transform, (im2.shape[1] + round(affine_transform[0][2]), im1.shape[0] + round(affine_transform[1][2])))

        # Place the first image onto the panorama
        panorama[:im1.shape[0], :im1.shape[1]] = im1

        #Proposed Change
        #panorama = cv2.warpAffine(im1, affine_transform, (im1.shape[1] + round(affine_transform[0][2])), (im1.shape[0] + round(affine_transform[1][2])))
        

    im1 = panorama
    img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)


cv2.imwrite(name+'.jpg', im1)