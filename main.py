import math
import cv2
import numpy as np

MIN_MATCH_COUNT = 10
FOCAL_LENGTH = 637.12


# Resize the Images to a predefined size
def resizeImages(image1, image2, target_size):
    resized_img1 = cv2.resize(image1, target_size, interpolation=cv2.INTER_AREA)
    resized_img2 = cv2.resize(image2, target_size, interpolation=cv2.INTER_AREA)
    return resized_img1, resized_img2


def featureDetection(image1, image2):
    # Apply SIFT on pictures to find key points and descriptors
    sift = cv2.SIFT_create()
    keypoints_1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors2 = sift.detectAndCompute(image2, None)

    return keypoints_1, descriptors1, keypoints_2, descriptors2


# Match descriptors to find pairs
def featureMatching(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    return matches


# Estimate the homography matrix from the key points and applying RANSAC
def homographyEstimation(keypoints_1, keypoints_2, good_matches):
    # Extract coordinates from key points
    src_pts = []
    dst_pts = []
    for m in good_matches:
        src_pts.append(keypoints_1[m.queryIdx].pt)
        dst_pts.append(keypoints_2[m.trainIdx].pt)

    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    # Get homography-matrix and apply RANSAC to filter datapoints
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, mask


# Calculate the camera intrinsics from focal length and image center
def calculateCameraIntrinsics(focal_length, image1, image2):
    cx = image1.shape[1] / 2
    cy = image2.shape[0] / 2
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

    return K


# Decompose the homography matrix to get the rotation matrices
def homographyDecomposition(H, K):
    _, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    return rotations, translations, normals


# Extract the euler angles from the rotation matrix
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if sy < 1e-6:
        singular = True
    else:
        singular = False

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


img1 = cv2.imread('IMG_2113.png')
img2 = cv2.imread('IMG_2115.png')

# Resize images to 800x600 format to avoid distortions
target_size = (800, 600)
img1, img2 = resizeImages(img1, img2, target_size)

# Change image to gray scale image for SIFT
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

keypoints_1, descriptors1, keypoints_2, descriptors2 = featureDetection(img1_gray, img2_gray)

# Match corresponding descriptors to find pairs
matches = featureMatching(descriptors1, descriptors2)

# Extract good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

if len(good_matches) >= MIN_MATCH_COUNT:

    H, mask = homographyEstimation(keypoints_1, keypoints_2, good_matches)

    # Calculate camera intrinsics
    # Focal length in px. Needs to be adjusted for each camera (Here Iphone12)
    K = calculateCameraIntrinsics(FOCAL_LENGTH, img1_gray, img2_gray)

    # Decompose homography matrix for rotations and translations
    rotations, translations, normals = homographyDecomposition(H, K)

    # Verify and select the correct rotation matrix
    for i, rotation_matrix in enumerate(rotations):
        eulerAngles = rotationMatrixToEulerAngles(rotation_matrix)
        pitch = np.degrees(eulerAngles[0])
        yaw = np.degrees(eulerAngles[1])
        roll = np.degrees(eulerAngles[2])

        print(f"Solution {i + 1}")
        print("Rotation matrix:\n", rotation_matrix)
        print("Pitch (X-axis rotation):", pitch)
        print("Yaw (Y-axis rotation):", yaw)
        print("Roll (Z-axis rotation):", roll)

else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, None,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('SIFT', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
