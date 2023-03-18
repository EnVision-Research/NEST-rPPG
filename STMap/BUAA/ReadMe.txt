# BUAA is proposed to evaluate the performance of the algorithm against various illumination.
# We only use data with illumination greater than or equal to 10 lux because underexposed images require special algorithms that are not considered in this article.


Handling procedure:
1. Mv_low_Light.py: Remove videos under 10 lux
2. Landmark.py: Get face landmarks from the video by using face_alignment (https://github.com/1adrianb/face-alignment)
3. Align_Face.py: Face alignment according to face landmarks
4. STMap.py: Generate a STMap (30FPS) with aligned faces and timestamps
5. Label_pro.py: Process the BVP signal