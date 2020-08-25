import cv2
import numpy as np

def answer1(x):
    
    if  0<=x<=90:
        ans ="a"
    if  90<x<=180:
        ans ="b"
    if  180<x<=270:
        ans ="c"
    if  270<x<=400:
        ans ="d"
        
    return ans


def answer2(x):
    
    if  0<=x<=144:
        ans ="a"
    if  144<x<=216:
        ans ="b"
    if  216<x<=288:
        ans ="c"
    if  288<x<=400:
        ans ="d"
        
    return ans        
def question(question): 
    question_array = [""]*40
    questionno = 0
    h,w = question.shape
    crop= question[15:h-15,15:w-15]
#    vis=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)   
#    print(question.shape)
#    h,w = crop.shape
#    plt.imshow(crop)
#    crop=cv2.equalizeHist(crop)
    th, im_th = cv2.threshold(crop,220,255,cv2.THRESH_BINARY_INV)
    
    
#    im_th+(np.uint8(im_th))
#    im_th1=~im_th
#    th, im_th2 = cv2.threshold(crop,20,255,cv2.THRESH_BINARY_INV)
    im_th0=im_th
    kernel = np.ones((5,5), np.uint8) 
    im_th0=cv2.morphologyEx(im_th0, cv2.MORPH_OPEN, kernel)
    
    im_th1=im_th
    im_th2=im_th1
    kernel = np.ones((1,60), np.uint8) 
    im_th1 = cv2.morphologyEx(im_th1, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((5,20), np.uint8) 
    im_th = cv2.erode(im_th,kernel,iterations = 1)
##    kernel = np.ones((5,5), np.uint8) 
#    im_th1 = cv2.morphologyEx(im_th1, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((5,100), np.uint8) 
#    im_th1 = cv2.erode(im_th1,kernel,iterations = 1)
#    kernel = np.ones((2,10), np.uint8) 
#    im_th1 = cv2.erode(im_th1,kernel,iterations = 1)
#    kernel = np.ones((10,40), np.uint8) 
    im_th1 = cv2.morphologyEx(im_th1, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5,w), np.uint8) 
    im_th1 = cv2.morphologyEx(im_th1, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((4,4), np.uint8) 
    im_th1 = cv2.morphologyEx(im_th1, cv2.MORPH_CLOSE, kernel)
    
    
#    cv2.imshow('mask.jpg',im_th1)
#    cv2.waitKey(0)
    
    
    kernel = np.ones((40,5), np.uint8) 
#    im_th2 = cv2.erode(im_th,kernel,iterations = 1)
##    kernel = np.ones((2,1), np.uint8) 
##    im_th = cv2.erode(im_th,kernel,iterations = 1)
##    kernel = np.ones((5,5), np.uint8) 
    im_th2 = cv2.morphologyEx(im_th2, cv2.MORPH_CLOSE, kernel)
#    kernel = np.ones((5,5), np.uint8) 
##    im_th2 = cv2.erode(im_th2,kernel,iterations = 1)
    kernel = np.ones((10,10), np.uint8) 
    im_th2 = cv2.erode(im_th2,kernel,iterations = 1)
    kernel = np.ones((h,5), np.uint8) 
    im_th2 = cv2.morphologyEx(im_th2, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((h,10), np.uint8) 
    im_th2 = cv2.morphologyEx(im_th2, cv2.MORPH_OPEN, kernel)
#    

    
    mycol=255*np.ones([crop.shape[0],crop.shape[1]]) 
#    
    (_,cnts22, _) = cv2.findContours(im_th2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    
    
#        print(len(cnts22))           
#    for z in range(0,crop.shape[1]):
#        st=sum(im_th2[:,z])
#    
#        if st<=4000:
#           
#            mycol[:,z]=np.zeros([crop.shape[0]])    
#      
    
            
     
    for z in range(0,crop.shape[0]):
        st=sum(im_th1[z,:])
    
        if st<=4000:
           
            mycol[z,:]=np.zeros([crop.shape[1]])
            
#    cv2.imshow('masdsdk.jpg',mycol)
#    cv2.waitKey(2)   
        
    (_,cnts, _) = cv2.findContours(np.uint8(mycol), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   
    for x in reversed(cnts):
#        print(x)
        col=crop[x[0][0][1]:x[2][0][1],x[0][0][0]:x[2][0][0]] 
#        print(col.shape)
        ret,thresh2 = cv2.threshold(col,20,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((18,18), np.uint8) 
        
        im_th4 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((5,5), np.uint8) 
        im_th4 = cv2.morphologyEx(im_th4, cv2.MORPH_CLOSE, kernel)
#        kernel = np.ones((5,5), np.uint8) 
#        im_th4 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
#        kernel = np.ones((20,20), np.uint8) 
#        im_th4 = cv2.morphologyEx(im_th4, cv2.MORPH_OPEN, kernel)
#        cv2.imshow('mask.jpg',im_th4)
#        cv2.waitKey(0)
#        
        (_,cnts, _) = cv2.findContours(im_th4, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        
#        print(len(cnts))
            
        if len(cnts) <= 0 :
            
            question_array[questionno] = "" 
           
# if cv2.contourArea(cnts[0])>950 or cv2.contourArea(cnts[0]) <600:
#                
        elif len(cnts) ==  1:                          
            
            M = cv2.moments(im_th4)
            
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
#            print((cX))  
            if len(cnts22)==4:
                question_array[questionno] = answer1(cX)
            else:
                question_array[questionno] = answer2(cX)    
#            
        elif len(cnts) ==  2:
            
            M1 = cv2.moments(cnts[1])
            cX1 = int(M1["m10"] / M1["m00"])
            cY1 = int(M1["m01"] / M1["m00"])
#            print(cX1)
            
            M2 = cv2.moments(cnts[0])
            cX2 = int(M2["m10"] / M2["m00"])
            cY2 = int(M2["m01"] / M2["m00"])
            
            if len(cnts22)==4:
                question_array[questionno] = answer1(cX1)+','+answer1(cX2)
            else:
                question_array[questionno] = answer2(cX1)+','+answer2(cX2)
                
            
        elif len(cnts) ==  3:
            
            M1 = cv2.moments(cnts[2])
            cX1 = int(M1["m10"] / M1["m00"])
            cY1 = int(M1["m01"] / M1["m00"])
        
            
            M2 = cv2.moments(cnts[1])
            cX2 = int(M2["m10"] / M2["m00"])
            cY2 = int(M2["m01"] / M2["m00"])
            
            M3 = cv2.moments(cnts[0])
            cX3 = int(M3["m10"] / M3["m00"])
            cY3 = int(M3["m01"] / M3["m00"])
            
            if len(cnts22)==4:
                question_array[questionno] = answer1(cX1)+','+answer1(cX2)+','+answer1(cX3)
            else:
                question_array[questionno] = answer2(cX1)+','+answer2(cX2)+','+answer2(cX3)
                    
        else:
            question_array[questionno]="a,b,c,d"
#        print(questionno+1,question_array[questionno])
        
        questionno=questionno+1
        
    return question_array          
        
#        thresh2 = cv2.GaussianBlur(col,(5,5),cv2.BORDER_DEFAULT) 
###            
#        binary = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
#            binary1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) 
             
#        (_,cnts, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#    orginal=cv2.imread("Aligned/Question.png",0)
    #cv2.imshow("Foreground", orginal)
    #cv2.waitKey(0)
    
#    th, im_th = cv2.threshold(orginal,127,255,0)
#    
#    im_th=~im_th
#    kernel = np.ones((4,4), np.uint8) 
#    binary = cv2.erode(im_th, kernel, iterations=2) 
#    
#    h,w = binary.shape                   
#    
#    question = 0
#    question_array = [""]*201
#    
#    for y in range(0, w,np.uint(np.floor(w/5))): 
#        
#        if (y +int(w/5-15) > w):
#            break 
#            
#        if y== 0:
#            column = binary[20:h-20, y+75: y +int(w/5-15)]
#            visc= orginal[20:h-20, y+75: y +int(w/5-15)]
#        
#        else:
#            column = binary[20:h-20, y+90: y +int(w/5-15)] 
#            visc= orginal[20:h-20, y+90: y +int(w/5-15)] 
#                          
#    #    cv2.imshow("Foreground", visc)
#    #    cv2.waitKey(0)
#    ##column = orginal[15:1696, 0:420] 
#    
#        for x in range(0, column.shape[0],np.uint(np.floor(h/40))):
#                    
#            if (x+40 > h):
#                break  
#            
#            question+=1
#            row = column[x:x+40,:] 
#            visr=visc[x:x+40,:]    
#    #        cv2.imshow("Foreground", visr)
#    #        cv2.waitKey(0)
#    
#            (_,cnts, _) = cv2.findContours(row, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
#            
#            
#            if len(cnts) <= 0 :
#                
#             question_array[question] = "" 
#               
#    # if cv2.contourArea(cnts[0])>950 or cv2.contourArea(cnts[0]) <600:
#    #                
#            elif len(cnts) ==  1:                          
#                
#                M = cv2.moments(row)
#                cX = int(M["m10"] / M["m00"])
#                cY = int(M["m01"] / M["m00"])
#                            
#                question_array[question] = answer(cX)
#    
#            elif len(cnts) ==  2:
#                
#                M1 = cv2.moments(cnts[1])
#                cX1 = int(M1["m10"] / M1["m00"])
#                cY1 = int(M1["m01"] / M1["m00"])
#            
#                
#                M2 = cv2.moments(cnts[0])
#                cX2 = int(M2["m10"] / M2["m00"])
#                cY2 = int(M2["m01"] / M2["m00"])
#                
#                question_array[question] = answer(cX1)+answer(cX2)
#                
#            elif len(cnts) ==  3:
#                
#                M1 = cv2.moments(cnts[2])
#                cX1 = int(M1["m10"] / M1["m00"])
#                cY1 = int(M1["m01"] / M1["m00"])
#            
#                
#                M2 = cv2.moments(cnts[1])
#                cX2 = int(M2["m10"] / M2["m00"])
#                cY2 = int(M2["m01"] / M2["m00"])
#                
#                M3 = cv2.moments(cnts[0])
#                cX3 = int(M3["m10"] / M3["m00"])
#                cY3 = int(M3["m01"] / M3["m00"])
#                
#                question_array[question] = answer(cX1)+answer(cX2)+answer(cX3)
#            else:
#                question_array[question]="a,b,c,d"
#    #            print(question,question_array[question])
#            
#    return question_array            
#
#
#            