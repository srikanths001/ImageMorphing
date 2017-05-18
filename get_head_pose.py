#!/usr/bin/env python

import cv2
import numpy as np
from imutils import face_utils
import dlib
import argparse

#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=False,
#        help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True,
#        help="path to input image")
#args = vars(ap.parse_args())

class GetHeadPose(object):
    def __init__(self):
        # 3D model points.
        self.model_points = np.array([
                                 (0.0, 0.0, 0.0),             # Nose tip
                                 (0.0, -330.0, -65.0),        # Chin
                                 (-225.0, 170.0, -135.0),     # Left eye left corner
                                 (225.0, 170.0, -135.0),      # Right eye right corne
                                 (-150.0, -150.0, -125.0),    # Left Mouth corner
                                 (150.0, -150.0, -125.0)      # Right mouth corner
                               ])
        self.image_points = np.array(self.model_points.shape)
        self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        self.camera_matrix = np.zeros((3,3))
        self.rotaton_vector = np.zeros((3,1))
        self.translation_vector = np.zeros((3,1))

    def getHeading(self, shape, image):
        #Image points
        self.image_points = np.array([
                                 shape[33],     # Nose tip #34
                                 shape[9],     # Chin #9
                                 shape[45],     # Left eye left corner #46
                                 shape[36],     # Right eye right corne #37
                                 shape[54],     # Left Mouth corner #55
                                 shape[48],      # Right mouth corner #49
                                 #shape[0],      #Left ear
                                 #shape[16]      #Right ear
                                ], dtype="double")
        #Compute approximate focal length
        im_size = image.shape
        focal_length = im_size[1]
        center = (im_size[1]/2, im_size[0]/2)
        self.camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
        #Solve PnP to get rotation and translation vector
        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return self.rotation_vector

    def projectPoint(self, image):
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), self.rotation_vector, self.translation_vector, self.camera_matrix, self.dist_coeffs)

        for p in self.image_points:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        p1 = ( int(self.image_points[0][0]), int(self.image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(image, p1, p2, (255,0,0), 2)
        return image

#if __name__=='__main__':
#    
