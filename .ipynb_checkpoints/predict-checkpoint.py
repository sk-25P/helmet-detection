import cv2
import PIL
from matplotlib import pyplot
from ultralytics import YOLO
model1=YOLO('best1.pt')
model2=YOLO('best (1).pt')
dict={}
k=0
temp_dict={}
count=0
def b_box_cord(values,dh,dw):
    
    lx=int((values[0]-values[2]/2)*dw)
    ly=int((values[1]-values[3]/2)*dh)
    rx=int((values[0]+values[2]/2)*dw)
    ry=int((values[1]+values[3]/2)*dh)
    if lx<0:
        lx=0
    if rx>dw-1:
        rx=dw-1
    if ly<0:
        ly=0
    if ry>dh-1:
        ry=dh-1
    return lx,ly,rx,ry

def predictor_two_wheeler(image):
    ans=True
    img=cv2.imread(image)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_copy=img.copy()
    result1=model1.predict(img,show_boxes=True)
    print(result1[0].boxes)
    two_wheel_box=result1[0].boxes.xywhn.tolist()
    vehicle_detect_cls=result1[0].boxes.cls.tolist()
    if vehicle_detect_cls.count(1)== 0:
        return True
    else:
        global dict
        global count
        count = vehicle_detect_cls.count(1)
        i=0
        for cls_prob,b_box in zip(vehicle_detect_cls,two_wheel_box):
            if cls_prob == 1:
                dh,dw,_=img.shape
                lx,ly,rx,ry=b_box_cord(b_box,dh,dw)
                crop_img=img_copy[ly:ry,lx:rx]
                dict[i]=crop_img
                i=i+1

                    
    return False
def predict_without_helmet(image):
    ans=True
    result2=model2.predict(image,show_boxes=True)
    cls_m2=result2[0].boxes.cls.tolist()
    print(result2[0].boxes)
    bounding_boxes=result2[0].boxes.xywhn.tolist()
    if cls_m2.count(1)==0:
        return True,image
    else:
        global k
        global temp_dict
        for cls_,b_box in zip(cls_m2,bounding_boxes):
            print(cls_)
            if cls_ == 1:
                ans=False
                dh,dw,_=image.shape
                lx,ly,rx,ry=b_box_cord(b_box,dh,dw)
                print(b_box)
                crop_img=image[ly:ry,lx:rx]
                temp_dict[k]=crop_img
                cv2.rectangle(image,(lx,ly),(rx,ry),(255,0,0),3)
                k=k+1

                
    return ans,image
def helper(i):
    global dict
    img=dict[i]
    return predict_without_helmet(img)
def helper2(q):
    global temp_dict
    im=temp_dict[q]
    return predict_without_helmet(im)
def select_head():
    return k
def select_vehicle():
    global count
    global dict
    return count,dict
bole=predictor_two_wheeler(r"datasets\Helmet_Detection-1\test\images\BikesHelmets89_png.rf.d1e18869738dcf1c6e436435046c5f99.jpg")
im=PIL.Image.open(r"datasets\Helmet_Detection-1\test\images\BikesHelmets89_png.rf.d1e18869738dcf1c6e436435046c5f99.jpg")
pyplot.imshow(im)
pyplot.show()