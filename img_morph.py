#!/usr/bin/env python

import cv2
import numpy as np
import dlib
from get_head_pose import GetHeadPose
import imutils
import argparse
import time

file_path="inputs_1.txt"
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_file", required=False,
        help="path to txt file containing images")
args = vars(ap.parse_args())

if args["input_file"] is not None:
    file_path=args["input_file"]
else:
    file_path="files.txt"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        #print(p)
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();
    #draw_delaunay(img, subdiv, delaunay_color )
    
    delaunayTri = []
    
    pt = []    
    
    count= 0    
    
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            count = count + 1 
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri,subdiv

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def getPoints(img):
    rects = detector(img, 1)
    shape = predictor(img, rects[0])
    shape = imutils.face_utils.shape_to_np(shape)
    (x, y, w, h) = imutils.face_utils.rect_to_bb(rects[0])
    rect=(x,y,w,h)
    left_ear_x = shape[16][0]+20
    left_ear_y = shape[16][1]
    right_ear_x = shape[0][0]-20
    right_ear_y = shape[0][1]
    head_right = (shape[0][0], shape[19][1]-30)
    head_left = (shape[16][0], shape[24][1]-30)
    #arr=np.array([[0,0],[img.shape[1]-5,0], [0,int(img.shape[0]-5)], [img.shape[1]-5,img.shape[0]-5],  [int(img.shape[1]/2),0], [0,int(img.shape[0]/2)], [img.shape[1]-5,int(img.shape[0]/2)], [int(img.shape[1]/2),int(img.shape[0]-1)], [int(img.shape[1]/2), img.shape[0]-50], [0,img.shape[0]-50], [img.shape[1]-5,img.shape[0]-50], [left_ear_x,left_ear_y], [right_ear_x,right_ear_y] ])
    arr=np.array([ [left_ear_x,left_ear_y], [right_ear_x,right_ear_y]  ])
    forhead = np.array([ [shape[19][0],shape[19][1]-30], [shape[24][0],shape[24][1]-30], [head_right[0],head_right[1]], [head_left[0],head_left[1]] ])
    #arr=np.array([[0,0],[0,img.shape[1]-5], [img.shape[0]-5,0], [img.shape[0]-5,img.shape[1]-5], [0,int(img.shape[1]/2)], [int(img.shape[0]/2), 0], [img.shape[0]-5,int(img.shape[1]/2)], [int(img.shape[0]/2), int(img.shape[1]/2)] ])
    shape = np.concatenate((shape,arr,forhead), axis=0)
    return shape,rect

# Get two good images
def getGoodImages(headings):
    sz = headings.shape[0]
    if(sz is None):
        return (0,0)
    else:
        #dist_from_zero=[]
        headings = abs(headings)
        #idx = np.argpartition(headings, sz-1)
        idx = np.argsort(headings)
        #print(idx)
        #return (idx[0], idx[1])
        return (idx[0], idx[sz-1])

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :

            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

if __name__=='__main__':
    #now = time.ctime(int(time.time()))
    start_time = time.time()
    print("Start morphing")
    #print("Start time: ") + str(now)
    with open(file_path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    if len(content) != 5:
        print("Enter path to five images in img_files.txt")
        #return -1
    else:
        get_head_orientation = GetHeadPose()
        images = np.empty(len(content), dtype=object)
        landmarks = np.empty(len(content), dtype=object)
        rects = np.empty(len(content), dtype=object)
        orientations = []
        for n in range(0, len(content)):
            images[n] = cv2.imread( content[n] )
            if images[n] is not None:
                landmarks[n], rects[n] = getPoints(images[n])
                #if landmarks[n] is not None:
                #    orientations[n] = 

        ## Get the head pose of each image
        headings = np.empty(len(landmarks))
        for i in range(0, len(images)):
            if landmarks[i] is not None:
                rotation_vector = get_head_orientation.getHeading(landmarks[i], images[i])
                #heading.append(rotation_vector[1][0])
                headings[i] = rotation_vector[0][0]
                #print(rotation_vector)

        good_imgs_idx = getGoodImages(headings)
        #print(good_imgs_idx)
        points=[]
        alpha = 0.5
        for i in range(0, landmarks[good_imgs_idx[0]].shape[0]):
            x = ( 1 - alpha ) * landmarks[good_imgs_idx[0]][i][0] + alpha * landmarks[good_imgs_idx[1]][i][0]
            y = ( 1 - alpha ) * landmarks[good_imgs_idx[0]][i][1] + alpha * landmarks[good_imgs_idx[1]][i][1]
            points.append((x,y))

        # Allocate space for final output
        imgMorph = np.zeros(images[good_imgs_idx[0]].shape, dtype = images[good_imgs_idx[0]].dtype)
        ##Get triangles
        sizeImg = images[good_imgs_idx[0]].shape
        rect = (0, 0, sizeImg[1], sizeImg[0])
        tris,subdiv = calculateDelaunayTriangles(rect, points)
        img_copy = images[good_imgs_idx[0]].copy()
        #draw_delaunay(img_copy, subdiv, [255,255,255] )

        for p in tris:
            x = int(p[0])
            y = int(p[1])
            z = int(p[2])

            t1 = [landmarks[good_imgs_idx[0]][x], landmarks[good_imgs_idx[0]][y], landmarks[good_imgs_idx[0]][z]]
            t2 = [landmarks[good_imgs_idx[1]][x], landmarks[good_imgs_idx[1]][y], landmarks[good_imgs_idx[1]][z]]
            t = [ points[x], points[y], points[z] ]

            # Morph one triangle at a time.
            morphTriangle(images[good_imgs_idx[0]], images[good_imgs_idx[1]], imgMorph, t1, t2, t, alpha)
        time_taken = time.time() - start_time
        print("Time taken: %.2f seconds" % time_taken)
        #cv2.imwrite("Morphed_Face.png", imgMorph)
        cv2.imshow("Morphed Face", np.uint8(imgMorph))
        #cv2.imshow("delaunay_triangle", np.uint8(img_copy))
        cv2.waitKey(0)
    print("Done")
