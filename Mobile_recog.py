import cv2
import numpy as np

def mobile(mob):

    orginal=np.uint8(mob)
    orginal=cv2.resize(orginal,(555,629))
#    cv2.imshow('a',orginal)
        
    h,w = orginal.shape
    crop= orginal[200:h-9,15:w-15]
    h1,w1 = crop.shape
    th, im_th0 = cv2.threshold(crop,220,255,0)
    im_th0=~im_th0
    kernel1= np.ones((5,30), np.uint8)  
    im_th0=cv2.morphologyEx(im_th0, cv2.MORPH_CLOSE, kernel1)
    
    kernel1=np.ones((5,30), np.uint8) 
    im_th0 = cv2.erode(im_th0, kernel1, iterations=1) 
    
    kernel1=np.ones((5,w1), np.uint8) 
    im_th0=cv2.morphologyEx(im_th0, cv2.MORPH_OPEN, kernel1)
    
    _,cnts0, _ = cv2.findContours(im_th0, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts0)==9:
        crop=orginal[150:h-9,15:w-15]
        h1,w1=crop.shape
    
    th, im_th = cv2.threshold(crop,127,255,0)
    im_th=~im_th
    kernel = np.ones((5,5), np.uint8) 
    number=[]
    binary = cv2.erode(im_th, kernel, iterations=2) 
#    cv2.imshow('a',binary)
#    cv2.waitKey(0)
    
    for y in range(0, w1,np.uint(np.floor(w1/10))):
        
        if (y +int(w1/10) > w1):
            break 
        column = binary[0:h1, y: y +int(w1/10)]
#        cv2.imshow('aa',column)
#        cv2.waitKey(0)
#        visc=crop[0:h1, y: y +int(w1/10)]
        countn = 0 
        for x in range(0, h1,np.uint(np.floor(h1/10))):
           
            if (x+int(h1/10) > h1):
                   break
            row = column[x:x+int(h1/10),:] 
#            visr=visc[x:x+int(h1/10),:] 
            countn+=1
            
#            cv2.imshow("Foreground", row)
#            cv2.waitKey(0)
            
            _,cnts, _ = cv2.findContours(row, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0: 
                number.append(str(countn))
    number=['0' if x == '10' else x for x in number]
    number=''.join(number)

    if len(number)>0:
        number=number
    else:
        number=""
#    print(number)
    return number

        

   