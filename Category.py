import cv2
import numpy as np

def category(cat):
    cate=""
    orginal=np.uint8(cat)
    orginal=cv2.resize(orginal,(202,626))
    c = ["GEN","OBC1","OBC2","SC","ST","PH"]

    h,w = orginal.shape
    crop= orginal[140:h-5,140:w-5]
#    cv2.imshow('l',crop)

    h1,w1 = crop.shape
    th, im_th = cv2.threshold(crop,127,255,0)

    im_th=~im_th
    kernel = np.ones((5,5), np.uint8) 
    binary = cv2.erode(im_th, kernel, iterations=2) 
    
    count = 0
    for x in range(0, h1,np.uint(np.floor(h1/6))):
           
        if (x+int(h1/6) > h1):
               break
        row = binary[x:x+int(h1/6),:] 
        visr=crop[x:x+int(h1/6),:] 
        count+=1
    #    cv2.imshow("Foreground", visr)
    #    cv2.waitKey(0)
        _,cnts, _ = cv2.findContours(row, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
        
        if len(cnts) == 1: 
            cate=c[count-1]               
    

    return cate