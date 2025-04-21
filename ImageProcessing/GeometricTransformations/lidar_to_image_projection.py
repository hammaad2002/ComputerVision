import os
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl # Needed for coloring
from zod import ZodFrames
from zod.data_classes import LidarData
from zod.constants import Camera, Lidar, Anonymization
from zod.visualization.lidar_on_image import visualize_lidar_on_image

dataset_root = os.path.expanduser("~/datasets/zod")  # your local path to zod
version = "mini"  # "mini" or "full"

# initialize ZodFrames
zod_frames = ZodFrames(dataset_root=dataset_root, version=version)

# get a single frame
zod_frame = zod_frames[62592]

# we can also get the calibrations
calibrations = zod_frame.calibration

# Get camera and lidar extrinsics and intrinsics and other info
camera_extrinsics = calibrations.cameras[Camera.FRONT].extrinsics.transform
lidar_extrinsics = calibrations.lidars[Lidar.VELODYNE].extrinsics.transform
camera_intrinsics = calibrations.cameras[Camera.FRONT].intrinsics
camera_distortion = calibrations.cameras[Camera.FRONT].distortion
image_size = calibrations.cameras[Camera.FRONT].image_dimensions

# Get frontal camera image
camera_core_frame = zod_frame.info.get_key_camera_frame(Anonymization.DNAT)
image = camera_core_frame.read()

# Get the lidar frame
lidar_core_frame = zod_frame.info.get_key_lidar_frame()
pc = lidar_core_frame.read()

# Get the timestamp of the keyframe
image_timestamp = zod_frame.info.keyframe_time.timestamp()

# Extract the top lidar only which is VLS-128
pc.points = pc.points[pc.diode_idx < 128]

# Calculate new optimal camera matrix for undistortion
new_camera_intrinsics = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    camera_intrinsics[:3, :3], camera_distortion, image_size, np.eye(3), balance=1.0  # balance=0.0: max FOV, 1.0: min black areas
)

# Undistort the image using the newK
undistorted_image = cv2.fisheye.undistortImage(
    image,
    camera_intrinsics[:3, :3],
    D=camera_distortion,
    Knew=new_camera_intrinsics,
    new_size=image_size,
)

# First transform the pointcloud to the camera frame
lidar_to_base_link_transform = lidar_extrinsics
base_link_to_camera_transform = np.linalg.inv(camera_extrinsics)
lidar_to_camera_transform = np.matmul(base_link_to_camera_transform, lidar_to_base_link_transform)
pointcloud_homogeneous = np.vstack((pc.points.T, np.ones((1, pc.points.shape[0]))))
pointcloud_in_camera_frame = np.matmul(lidar_to_camera_transform, pointcloud_homogeneous)

# Either do the above or more verbose way
transform_pointcloud_to_baselink = np.matmul(lidar_to_base_link_transform, pointcloud_homogeneous) # we have reached baselink
transform_pointcloud_to_camera_frame = np.matmul(base_link_to_camera_transform, transform_pointcloud_to_baselink) # from baselink we go to camera frame

# Proof that both are equal :)
assert np.isclose(transform_pointcloud_to_camera_frame, pointcloud_in_camera_frame).all(), "This should match!"

# Normalize homogeneous coordinates (divide by w)
# w is typically 1 for standard transformations, but good practice to include
w = (pointcloud_in_camera_frame.T)[:, 3:]
points_cam_3d = (pointcloud_in_camera_frame.T)[:, :3] / w # Nx3 -> [X, Y, Z] in camera frame


pointcloud_in_image_plane = np.matmul(new_camera_intrinsics, points_cam_3d.T)
pointcloud_in_image_plane[:-1, :] = pointcloud_in_image_plane[:-1, :] / pointcloud_in_image_plane[-1, :]
xyd = pointcloud_in_image_plane

frontal_undistorted_frame = copy.deepcopy(undistorted_image)
for x, y in zip(xyd[0], xyd[1]):
    cv2.circle(frontal_undistorted_frame, (int(x), int(y)), 2, (0, 0, 255), -1)

plt.imshow(frontal_undistorted_frame)
plt.show()