import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from zod import ZodFrames
from zod.constants import Camera, Lidar

path_to_zod = os.path.expanduser("~/datasets/zod")
version = "mini"
zod_frames = ZodFrames(dataset_root=path_to_zod, version=version)
zod_frame = zod_frames[87912]

calibrations = zod_frame.calibration
lidar_extrinsics = calibrations.lidars[Lidar.VELODYNE].extrinsics.transform
camera_intrinsics = calibrations.cameras[Camera.FRONT].intrinsics
camera_extrinsics = calibrations.cameras[Camera.FRONT].extrinsics.transform
camera_distortion = calibrations.cameras[Camera.FRONT].distortion
camera_undistortion = calibrations.cameras[Camera.FRONT].undistortion
camera_image_size = calibrations.cameras[Camera.FRONT].image_dimensions
camera_field_of_view = calibrations.cameras[Camera.FRONT].field_of_view

camera_core_frame = zod_frame.info.get_camera_frames()[0]
image= camera_core_frame.read()

TL_SRC = [1634, 1132]
TR_SRC = [2027, 1127]
BR_SRC = [2617, 1587]
BL_SRC = [25, 1546] #
OUTPUT_IMAGE_WIDTH = 400
OUTPUT_IMAGE_HEIGHT = 400

source_points = np.float32([TL_SRC, TR_SRC, BR_SRC, BL_SRC])

destination_points = np.float32([
    [0, 0],                                    # Top Left
    [OUTPUT_IMAGE_WIDTH, 0],                   # Top Right
    [OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT], # Bottom Right
    [0, OUTPUT_IMAGE_HEIGHT]                   # Bottom Left
])

output_size = (OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)

M = cv2.getPerspectiveTransform(source_points, destination_points)
warped_image = cv2.warpPerspective(image, M, output_size)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(warped_image)
plt.show()