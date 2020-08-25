# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:40:17 2020

@author: Koshy G
"""


import os
import csv
import Question_recog
import segments
import Name_recog
import Mobile_recog
import Roll_no
import Center_code
import Category



In_path ="Input/"
Score_path = "Score/"
valid_images = [".jpg",".gif",".png",".tga"]
data=[]
data.append(['Centre Code','Roll No','Name','Category','Mobile Number','question'])



#
i=0
for s in os.listdir(In_path):
    
        print(i)
        i=i+1
        ext = os.path.splitext(s)[1]
        if ext.lower() not in valid_images:
            continue
        input_img=os.path.join(In_path,s)

        q1,q2,q3,q4,q5,name,mob,roll,center,cat = segments.segment(input_img)
        quest1=Question_recog.question(q1)
        quest2=Question_recog.question(q2)
        quest3=Question_recog.question(q3)
        quest4=Question_recog.question(q4)
        quest5=Question_recog.question(q5)
        quest=quest1+quest2+quest3+quest4+quest5
        name = Name_recog.nameread(name)
        mobile = Mobile_recog.mobile(mob)
        rollno = Roll_no.roll_no(roll)
        code=Center_code.code(center)
        cate = Category.category(cat)
        
        data.append([code,rollno,name,cate,mobile,quest])
       
        
       
with open('output.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(data)
            
       
