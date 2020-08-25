import cv2
import numpy as np

def code(center):
    
    orginal=np.uint8(center)
    orginal=cv2.resize(orginal,(267,583))

    h,w = orginal.shape
    crop= orginal[158:h-5,18:w-18]
    
    
    th, im_th = cv2.threshold(crop,127,255,0)
    im_th=~im_th
    kernel = np.ones((5,5), np.uint8) 
    binary = cv2.erode(im_th, kernel, iterations=2) 
    
    h1,w1 = crop.shape
    code = []
    for y in range(0, w1,np.uint(np.floor(w1/4))): 
        if (y +int(w1/4) > w1):
            break 
        column = binary[0:h1, y: y +int(w1/4)]
        visc=crop[0:h1, y: y +int(w1/4)]
        count=0          
        for x in range(0, h1,np.uint(np.floor(h1/10))):
               
            if (x+int(h1/10) > h1):
                   break
            row = column[x:x+int(h1/10),:] 
            visr=visc[x:x+int(h1/10),:] 
            count+=1
            
    #        cv2.imshow("Foreground", visr)
    #        cv2.waitKey(0)
            
            _,cnts, _ = cv2.findContours(row, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)       
            if len(cnts) == 1: 
                code.append(str(count))
    if len(code)>0:
        code=['0' if x == '10' else x for x in code]
        code=''.join(code)
        code=int(code)
    else:
        code=""
    return code

            