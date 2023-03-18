# VIPL-HR have nine scenarios, three RGB cameras, different illumination conditions, and different levels of movement.
# It is worth mentioning that the BVP signal and video of the dataset do not match in the time dimension.
# Therefore, it is recommended that VIPL should not use BVP signals as supervision when training with other data sets
# We normalize STMap to 30 fps by cubic spline interpolation to solve the problem of the unstable frame rate of video.

Handling procedure:
1. Landmark.py: Get face landmarks from the video by using face_alignment (https://github.com/1adrianb/face-alignment)
2. Landmark_proce.py: The abnormal landmarks are interpolated using sequential continuity.
3. Align_Face.py: Face alignment according to face landmarks
4. STMap.py: Generate a STMap (30FPS) with aligned faces and timestamps