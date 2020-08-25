import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as skm

alpha =['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


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
    
    
    
    
    

def nameread(name):    
    orginal=name 
#    cv2.imwrite("Foreground2.jpg", orginal )
#    cv2.waitKey(0)

    h,w = orginal.shape
    crop= orginal[150:h-10,15:w-10]
    vis=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)   

    h,w = crop.shape
#    plt.imshow(crop)
#    crop=cv2.equalizeHist(crop)
    th, im_th = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    im_th+(np.uint8(im_th))
    im_th1=~im_th
    im_th2=im_th1
    im_th0=im_th2
    kernel = np.ones((20,20), np.uint8) 
    im_th0=cv2.morphologyEx(im_th0, cv2.MORPH_OPEN, kernel)
    
    im_th1=~im_th
    im_th2=im_th1
    kernel = np.ones((5,20), np.uint8) 
#    im_th = cv2.erode(im_th,kernel,iterations = 1)
#    kernel = np.ones((2,1), np.uint8) 
#    im_th = cv2.erode(im_th,kernel,iterations = 1)
#    kernel = np.ones((5,5), np.uint8) 
    im_th1 = cv2.morphologyEx(im_th1, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((2,30), np.uint8) 
    im_th1 = cv2.erode(im_th1,kernel,iterations = 1)
    kernel = np.ones((2,10), np.uint8) 
    im_th1 = cv2.erode(im_th1,kernel,iterations = 1)
    kernel = np.ones((10,10), np.uint8) 
    im_th1 = cv2.morphologyEx(im_th1, cv2.MORPH_OPEN, kernel)
    
    
    
    kernel = np.ones((10,2), np.uint8) 
#    im_th = cv2.erode(im_th,kernel,iterations = 1)
#    kernel = np.ones((2,1), np.uint8) 
#    im_th = cv2.erode(im_th,kernel,iterations = 1)
#    kernel = np.ones((5,5), np.uint8) 
    im_th2 = cv2.morphologyEx(im_th2, cv2.MORPH_CLOSE, kernel)
#    kernel = np.ones((30,5), np.uint8) 
#    im_th2 = cv2.erode(im_th2,kernel,iterations = 1)
#    kernel = np.ones((10,2), np.uint8) 
#    im_th2 = cv2.erode(im_th2,kernel,iterations = 1)
    kernel = np.ones((10,10), np.uint8) 
    im_th2 = cv2.morphologyEx(im_th2, cv2.MORPH_OPEN, kernel)
    

    
    mycol=255*np.ones([crop.shape[0],crop.shape[1]]) 
#    
    
               
    for z in range(0,crop.shape[1]):
        st=sum(im_th2[:,z])
    
        if st<=4000:
           
            mycol[:,z]=np.zeros([crop.shape[0]])    
            
            
     
    for z in range(0,crop.shape[0]):
        st=sum(im_th1[z,:])
    
        if st<=4000:
           
            mycol[z,:]=np.zeros([crop.shape[1]])
            
#    cv2.imwrite('mask.jpg',mycol)
    
    
#    _,contours,_ = cv2.findContours(np.uint8(mycol),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   
#    
#    for i in range(0,len(contours),26):
##        count=0
#        c=sort_contours(contours[i:i + 26],method="top-to-bottom")
#        for c1 in c[0]:
#            
#            mask = np.zeros(im_th.shape, dtype="uint8")
#            cv2.drawContours(mask, [c1], -1, 255, -1)
#            cv2.drawContours(vis, c1, -1, 255, 3)
#            cv2.imshow("For",cv2.resize( mask,(int(w/2),int(h/2) )))
#            cv2.waitKey(0)
##                

#    out1=mycol*imth1;
#            
#    kernel = np.ones((2,2), np.uint8) 
#    im_th = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
#    im_th=skm.skeletonize_3d(im_th)
    
    
#    for i in range (0,h):
#        if(sum(im_th[i,:])<10000):
#            im_th[i,:]=0
#    
#    for j in range (0,w):
#        if(sum(im_th[:,j])<10000):
#            im_th[:,j]=0
#    
   
    
    
#    
    _,contours,_ = cv2.findContours(np.uint8(mycol),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
#    print(len(contours))
    
    valid_contours=[]
    
    for c in reversed(contours):
        (x, y, w1, h1) = cv2.boundingRect(c)
#        
#        print(x)
#        col=im_th0[x[0][0][1]:x[2][0][1],x[0][0][0]:x[2][0][0]] 
#        
#        cv2.imshow("Foreground2.jpg",np.uint8( col) )
#        cv2.waitKey(0)
        ar = w1 / float(h1)
#        print(w1,h1,ar)
     
        if w1 >= 20 and h1 >= 20:# and ar >= 0.9 and ar <= 1.5:
            valid_contours.append(c)
    
    valid_contours,_ = sort_contours(valid_contours,method="left-to-right")
    
#    cv2.drawContours(crop, valid_contours[0:26], -1, (0,255,0), 3)
#    cv2.imshow("For",cv2.resize( crop,(int(w/2),int(h/2) )))
#    cv2.waitKey(0)

#    for x in valid_contours:
#        cv2.drawContours(crop, x, -1, (0,255,0), 3)
#        cv2.imshow("For",cv2.resize( crop,(int(w/2),int(h/2) )))
#        cv2.waitKey(0)
    dic=[]
    for i in range(0,len(valid_contours),26):
        count=0
        c=sort_contours(valid_contours[i:i + 26],method="top-to-bottom")
        flag=0
        for c1 in c[0]:
            
            mask = np.zeros(im_th.shape, dtype="uint8")
            cv2.drawContours(mask, [c1], -1, 255, -1)
            cv2.drawContours(vis, c1, -1, 255, 3)
            

            
            
            mask = cv2.bitwise_and( np.uint8(mycol),im_th0, mask=mask)
#            mask=mask*mycol
            count+=1

            total = cv2.countNonZero(mask)
            if total>20:
                flag=1
#            print(total,count)
            dic.append([total,count])
        if flag==0:
            dic.append([27,27])
        
#    cv2.imshow('dd',im_th0)
#    cv2.waitKey(0)
    score=max(dic)
   
    result=[]
#    flag=0
#    noc=0
    for a in dic:
#        print(noc,flag)
#        noc=noc+1
        if a[0]>score[0]-int(score[0]/3):
#            print(alpha[a[1]-1])
            result.append(alpha[a[1]-1])
#            flag=1
        if a[1]==27:
            result.append(' ')
#        else:           
##            print(noc)
#                
#            if noc==26 and flag==0:
#                result.append(' ')  
#                noc=0
#            if noc==26 and flag==1:
#                flag=0
#                noc=0
#            flag=1
##        else:
##            noc=noc+1
#        
##        print(noc,flag)
##        if a[1]==26:           
###           noc=0
#        if flag==1: 
#            flag=0
#            print(flag)
#        else:    
#           
            
#        else:
            
        
            
           
#    print(result)
    name=result
#            name=str(name).split()
    name=''.join(name)  
    name=str(name).split()   
    name=' '.join(name)  
#    print(name)
#    name.lstrip()
#    return name 
  
    return name       

#    
#    
#    
