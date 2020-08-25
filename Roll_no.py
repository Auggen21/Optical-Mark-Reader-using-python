import cv2
import numpy as np

def roll_no(roll):
    
    orginal=np.uint8(roll)
    orginal=cv2.resize(orginal,(487,581))
    h,w = orginal.shape
    crop= orginal[165:h-15,25:w-25]
    
    th, im_th = cv2.threshold(crop,127,255,0)
    im_th=~im_th
    kernel = np.ones((5,5), np.uint8) 
    binary = cv2.erode(im_th, kernel, iterations=2) 
    
    h1,w1 = crop.shape
    
    roll_no = []
    for y in range(0, w1,np.uint(np.floor(w1/7))): 
        if (y +int(w1/7) > w1):
            break 
        column = binary[0:h1, y: y +int(w1/7)]
        visc=crop[0:h1, y: y +int(w1/7)]
        
        count=0
        for x in range(0, h1,np.uint(np.floor(h1/10))):
               
            if (x+int(h1/10) > h1):
                   break
            row = column[x:x+int(h1/10),:] 
            visr=visc[x:x+int(h1/10),:] 
            count+=1
#            cv2.imshow("Foreground", row)
#            cv2.waitKey(0)
            _,cnts, _ = cv2.findContours(row, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
            if len(cnts) > 0:
                if cv2.contourArea(cnts[0])>80:

                    roll_no.append(str(count))
            
    #        cv2.imshow("Foreground", visr )
    #        cv2.waitKey(0)
    if len(roll_no)>0:
        roll_no=['0' if x == '10' else x for x in roll_no]
        roll_no=''.join(roll_no)
        roll_no=int(roll_no)
    else:
        roll_no=""
    return roll_no
