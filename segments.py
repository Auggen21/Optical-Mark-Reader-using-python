import numpy as np
import cv2
import cv2.aruco as aruco
import os
import matplotlib.pyplot as plt

#https://pysource.com/2018/02/14/perspective-transformation-opencv-3-4-with-python-3-tutorial-13/

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
#	print(dst)
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped 

 
 
def sort_contours(cnts, method="top-to-bottom"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

    
def segment(path):
   
    valid_images = [".jpg",".gif",".png",".tga"]

#    for s in os.listdir(path):
#        ext = os.path.splitext(s)[1]
#        if ext.lower() not in valid_images:
#            continue
#        imagec=cv2.imread(os.path.join(path,s))
     
    imagec=cv2.imread(path)   
#    imagec=cv2.imread('test/1.jpg') 
    imagec=cv2.resize(imagec,(2500,3500))
    
    image=cv2.cvtColor(imagec,cv2.COLOR_BGR2GRAY)
#    cv2.imshow('vaa',image)
    
    h,w = image.shape
    
    image = cv2.blur(image,(3,3))
    im2 = np.zeros(image.shape)
    im2=np.uint8(im2)    
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        image, aruco_dict, parameters=parameters)
        
#    frame_markers=aruco.drawDetectedMarkers(image, corners, ids, borderColor=(0, 220, 0))
#    aruco.drawDetectedMarkers(image, rejectedImgPoints, borderColor=(0, 0, 0))
#    cv2.imshow('00g',cv2.resize(image,(600,600)))
    
    for rejected in rejectedImgPoints:
         rejected = rejected.reshape((4, 2))
         
    #     cv2.line(image, tuple(rejected[0]), tuple(rejected[1]), (0, 0, 255), thickness=2)
    #     cv2.line(image, tuple(rejected[1]), tuple(rejected[2]), (0, 0, 255), thickness=2)
    #     cv2.line(image, tuple(rejected[2]), tuple(rejected[3]), (0, 0, 255), thickness=2)
    #     cv2.line(image, tuple(rejected[3]), tuple(rejected[0]), (0, 0, 255), thickness=2)
    #     cv2.rectangle(image, (rejected[0][0],rejected[0][1]), (rejected[2][0],rejected[2][1]), (255, 255, 255), -1) 
    #     cv2.rectangle(im2, (rejected[0][0],rejected[0][1]), (rejected[2][0],rejected[2][1]), (255, 255, 255), -1) 
         penta = np.array([[rejected[0],rejected[1],rejected[2],rejected[3]]], np.int32)
         area=(cv2.contourArea(penta))
         if area>(h*w/1000):  
             cv2.polylines(im2 ,penta,True, (255, 255, 255), 1)
             cv2.fillPoly(im2,penta, 255)     
             
             
   
#    cv2.imshow('g',cv2.resize(im2,(800,800)))
    #cv2.waitKey(0)
    #cv2.imwrite('diagonal1.png',im2)
    
    temp=im2.copy() 
    image2=image.copy()     
          
    while True:
        
        quater=int(h/3)
        for t in range (quater,h-quater):
            if np.sum(temp[t,:])==0:
                middle = t
                break

    
        top=temp[:middle,:]
        btm=temp[middle:,:]
    
        top_img=image2[:middle,:]
        btm_img=image2[middle:,:]
    
#        cv2.imshow('g',cv2.resize(btm,(600,600)))
#        cv2.waitKey(0)
    
        (_,ct,_) = cv2.findContours(top, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        (_,cb,_) = cv2.findContours(btm, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        a1= sorted(ct, key=lambda x: cv2.contourArea(x))
        a2= sorted(cb, key=lambda x: cv2.contourArea(x))
        
        if (cv2.contourArea(a1[-1])>cv2.contourArea(a2[-1])):
            break
        
        else:
            center = (w / 2, h / 2) 
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            rotatedim2 = cv2.warpAffine(im2, M, (w, h))
            rotatedimage = cv2.warpAffine(image2, M, (w, h))
            
            temp=rotatedim2.copy()
            image2=rotatedimage.copy()            
#            cv2.imshow('va',cv2.resize(rotatedimage,(600,600)))
            continue
        
#    cv2.imshow('va',cv2.resize(btm_img,(600,600)))

            

    ctSort = sorted(ct, key=lambda x: cv2.contourArea(x))
    cbSort,b=sort_contours(cb, "right-to-left")
    
    top_seg=[]
    for x2 in reversed(ctSort):
        
    #    cv2.drawContours(top_img, x2, -1, (0,0,0), 10)
    #    cv2.imshow('output', cv2.resize(top_img,(800,800)))            
        rc = cv2.minAreaRect(x2)
        box = cv2.boxPoints(rc)
    #    box=np.uint32(box)
#        for p in cord:
#            pt = (p[0],p[1])
    #        cv2.circle(top_img,pt,5,(0,255,0),15)
    
        warped = four_point_transform(top_img, box)
        top_seg.append(warped)     
    
    cat=top_seg[4]
    center=top_seg[3]
    roll=top_seg[2]
    mob=top_seg[1]
    name=top_seg[0]
    
    
    btm_seg=[]
    
    for x3 in reversed(cbSort):
        
    #    cv2.drawContours(top_img, x2, -1, (0,0,0), 10)
    #    cv2.imshow('output', cv2.resize(top_img,(800,800)))            
        rc = cv2.minAreaRect(x3)
        box1 = cv2.boxPoints(rc)
    #    box=np.uint32(box)
#        for p1 in cord:
#            pt1 = (p1[0],p1[1])
    #        cv2.circle(top_img,pt,5,(0,255,0),15)
        warped1 = four_point_transform(btm_img, box1)
        btm_seg.append(warped1)     
    
    q5=btm_seg[4]
    q4=btm_seg[3]
    q3=btm_seg[2]
    q2=btm_seg[1]
    q1=btm_seg[0]
    
    #
    #cv2.imshow('output', q1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



    return q1,q2,q3,q4,q5,name,mob,roll,center,cat