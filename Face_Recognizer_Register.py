import cv2
import numpy as np
import csv

#face Cascade
cascade_path="cascades\haarcascade_frontalface_alt.xml"
cascade=cv2.CascadeClassifier(cascade_path)
#eye cascade
cascade_path_eye="cascades\haarcascade_eye.xml"
cascade_eye=cv2.CascadeClassifier(cascade_path_eye)
#nose cascade
cascade_path_nose="cascades\haarcascade_mcs_nose.xml"
cascade_nose=cv2.CascadeClassifier(cascade_path_nose)
#mouth cascade
cascade_path_mouth="cascades\haarcascade_mcs_mouth.xml"
cascade_mouth=cv2.CascadeClassifier(cascade_path_mouth)

Green=(0,255,0)
Black=0
White=255

#take a picutre by Web Camera
cap=cv2.VideoCapture(0)
while(1):
    ret,image=cap.read()
    cv2.imshow("Camera Test",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break

#trim the face part from the picture
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
facerect=cascade.detectMultiScale(gray,1.3,5)
if len(facerect)>0:
    for (x,y,w,h) in facerect:
        face_part=image[y:y+h,x:x+w]
        face_part_gray=gray[y:y+h,x:x+w]
        cv2.imwrite("Face_part.jpg",face_part_gray)
        cv2.imshow("Face_part.jpg",face_part_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    print("Couldn't find any face. Try one more time!")
    exit()

#to make a picture high contrast（sigmoid function）
#for keeping the calculation faster, constract sigmoid only with universal function
def sigmoid(image,a):
    shape=image.shape
    ones=np.ones(shape)
    mid=np.mean([np.min(image),np.max(image)])
    img_sigmoid=1+np.exp(-a*(image-mid))
    img_sigmoid=np.divide(ones,img_sigmoid)*255
    img_sigmoid=np.rint(img_sigmoid)
    img_contrast=img_sigmoid.astype(np.uint8)
    return img_contrast

#sigmoid for nose
def sigmoid_nose(image,a):
    shape=image.shape
    ones=np.ones(shape)
    min=np.min(image)
    img_sigmoid=1+np.exp(-a*(image-1.4*min))
    img_sigmoid=np.divide(ones,img_sigmoid)*255
    img_sigmoid=np.rint(img_sigmoid)
    img_contrast=img_sigmoid.astype(np.uint8)
    return img_contrast

#Canny edge
def Canny(image,a,b):
    image_edge=cv2.Canny(image,threshold1=a,threshold2=b)
    return image_edge

#function which find a far right/left black/white point in a picture
#put Black or White into "color"
def right_corner(part,color):
    index=np.where(part==color)
    Y=np.array(index[0])
    X=np.array(index[1])
    Z=np.vstack((X,Y))
    order=Z[0,:].argsort()
    order=np.take(Z,order,1)
    corner_right=np.array([order[0,0],order[1,0]])
    return corner_right

def left_corner(part,color):
    index=np.where(part==color)
    Y=np.array(index[0])
    X=np.array(index[1])
    Z=np.vstack((X,Y))
    order=Z[0,:].argsort()
    order=np.take(Z,order,1)
    corner_left=np.array([order[0,-1],order[1,-1]])
    return corner_left


#find eye parts
#left/right_eye_lower is for omitting eyebrows
eye=cascade_eye.detectMultiScale(face_part_gray,1.3,5)
if len(eye)>0:
    if eye[0,0]>eye[1,0]:
        left_eye=np.array([eye[0]])
        for (x,y,w,h) in left_eye:
            left_eye_gray=face_part_gray[y:y+h,x:x+w]
            Y=y+int(h/2)
            left_eye_lower=face_part_gray[Y:y+h,x:x+w]
            left_eye_area=np.array([x,Y])

        right_eye=np.array([eye[1]])
        for (x,y,w,h) in right_eye:
            right_eye_gray=face_part_gray[y:y+h,x:x+w]
            Y=y+int(h/2)
            right_eye_lower=face_part_gray[Y:y+h,x:x+w]
            right_eye_area=np.array([x,Y])

    else:
        left_eye=np.array([eye[1]])
        for (x,y,w,h) in left_eye:
            left_eye_gray=face_part_gray[y:y+h,x:x+w]
            Y=y+int(h/2)
            left_eye_lower=face_part_gray[Y:y+h,x:x+w]
            left_eye_area=np.array([x,Y])

        right_eye=np.array([eye[0]])
        for (x,y,w,h) in right_eye:
            right_eye_gray=face_part_gray[y:y+h,x:x+w]
            Y=y+int(h/2)
            right_eye_lower=face_part_gray[Y:y+h,x:x+w]
            right_eye_area=np.array([x,Y])

else:
    print("Couldn't find any eye. Please try one more time!")
    exit()

#High contrast
left_lower_contrast=sigmoid(left_eye_lower,10)
right_lower_contrast=sigmoid(right_eye_lower,10)
#make edge by Canny
left_eye_canny=Canny(left_lower_contrast,30,70)
right_eye_canny=Canny(right_lower_contrast,30,70)

cv2.imshow("1.jpg",right_eye_gray)
cv2.imshow("2.jpg",left_eye_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("3.jpg",right_lower_contrast)
cv2.imshow("4.jpg",left_lower_contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("5.jpg",right_eye_canny)
cv2.imshow("6.jpg",left_eye_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

#empty datas
data={}
list=[]
abscissa_eye=[]
abscissa_nose=[]

#right corner point of right eye
right_eye_right=right_corner(right_eye_canny,White)
right_eye_rc=right_eye_area+right_eye_right
abscissa_eye.append(right_eye_rc[1])
data["right_eye_rc"]=right_eye_rc
point_right_eye_rc=(right_eye_rc[0],right_eye_rc[1])
list.append(point_right_eye_rc)

#left corner point of right eye
right_eye_left=left_corner(right_eye_canny,White)
right_eye_lc=right_eye_area+right_eye_left
abscissa_eye.append(right_eye_lc[1])
data["right_eye_lc"]=right_eye_lc
point_right_eye_lc=(right_eye_lc[0],right_eye_lc[1])
list.append(point_right_eye_lc)

#right corner point of left eye
left_eye_right=right_corner(left_eye_canny,White)
left_eye_rc=left_eye_area+left_eye_right
abscissa_eye.append(left_eye_rc[1])
data["left_eye_rc"]=left_eye_rc
point_left_eye_rc=(left_eye_rc[0],left_eye_rc[1])
list.append(point_left_eye_rc)

#left corner point of left eye
left_eye_left=left_corner(left_eye_canny,White)
left_eye_lc=left_eye_area+left_eye_left
abscissa_eye.append(left_eye_lc[1])
data["left_eye_lc"]=left_eye_lc
point_left_eye_lc=(left_eye_lc[0],left_eye_lc[1])
list.append(point_left_eye_lc)


#find a nose below the eyes
abscissa_eye=int(np.mean(abscissa_eye))
nose_area_gray=face_part_gray[abscissa_eye:,:]
nose_area=np.array([0,abscissa_eye])

#find nose part
nose_rect=cascade_nose.detectMultiScale(nose_area_gray,1.3,5)
if len(nose_rect)>0:
    for (x,y,w,h) in nose_rect:
        nose_gray=nose_area_gray[y:y+h,x:x+w]
        nose_rect=np.array([nose_rect[0,0],nose_rect[0,1]])

else:
    print("Couldn't find any nose. Pease try one more time!")
    exit()

#High contrast, Canny
nose_contrast=sigmoid_nose(nose_gray,10)
nose_canny=Canny(nose_contrast,30,70)

cv2.imshow("7.jpg",nose_gray)
cv2.imshow("8.jpg",nose_contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("9.jpg",nose_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

#right nostril
nose_rc=right_corner(nose_canny,White)
nose_right=nose_area+nose_rect+nose_rc
abscissa_nose.append(nose_right[1])
data["nose_right"]=nose_right
point_nose_right=(nose_right[0],nose_right[1])
list.append(point_nose_right)

#left nostril
nose_lc=left_corner(nose_canny,White)
nose_left=nose_area+nose_rect+nose_lc
data["nose_left"]=nose_left
abscissa_nose.append(nose_left[1])
point_nose_left=(nose_left[0],nose_left[1])
list.append(point_nose_left)

#find a mouth under a nose
abscissa_nose=int(np.mean(abscissa_nose))
mouth_area_gray=face_part_gray[abscissa_nose:,:]
mouth_area=np.array([0,abscissa_nose])

#find mouth
mouth_rect=cascade_mouth.detectMultiScale(mouth_area_gray,1.3,5)
if len(mouth_rect)>0:
    for (x,y,w,h) in mouth_rect:
        mouth_gray=mouth_area_gray[y:y+h,x:x+w]
        mouth_rect=np.array([mouth_rect[0,0],mouth_rect[0,1]])

else:
    print("Couldn't find any nose. Please try one more time!")

#High contrast, Canny
mouth_canny=Canny(mouth_gray,100,200)

cv2.imshow("10.jpg",mouth_gray)
cv2.imshow("11.jpg",mouth_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

#right corner of mouth
mouth_rc=right_corner(mouth_canny,White)
mouth_right=mouth_area+mouth_rect+mouth_rc
data["mouth_right"]=mouth_right
point_mouth_right=(mouth_right[0],mouth_right[1])
list.append(point_mouth_right)

#left corner of mouth
mouth_lc=left_corner(mouth_canny,White)
mouth_left=mouth_area+mouth_rect+mouth_lc
data["mouth_left"]=mouth_left
point_mouth_left=(mouth_left[0],mouth_left[1])
list.append(point_mouth_left)


#plot all points
for i in list:
        cv2.circle(face_part,i,1,Green,2)

cv2.imshow("face_pointed",face_part)
cv2.waitKey(0)
cv2.destroyAllWindows()


#base point. center of eye corners
center_point=(left_eye_lc+right_eye_rc)/2
#base distance
D=left_eye_lc-right_eye_rc
D=D[0]**2+D[1]**2
base_distance=np.absolute(np.sqrt(D))

#function for calculation the distance between base point and each point
def distance(point):
    d=point-center_point
    d=d[0]**2+d[1]**2
    D=np.absolute(np.sqrt(d))
    return D

#put distances into dictionary
for i in data.keys():
    data[i]=distance(data[i])/base_distance

print(data)

#save the dictionary as a csv file
file_name=input("Type your name to register. Please save as csv file...")
while file_name[-4:]!=".csv":
    file_name=input("File type error! Please save as csv file...")

f=open(file_name,"w")
header=data.keys()
title={}
for i in header:
    title[i]=i

data_set=[title,data]
writer=csv.DictWriter(f,header)
writer.writerows(data_set)
f.close()
