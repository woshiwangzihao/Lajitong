######## Picamera Object Detection Using Tensorflow Classifier #########
# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import RPi.GPIO as GPIO
from omxplayer import OMXPlayer
from pathlib import Path
from time import sleep
import time
#set PWM constants
GPIO.setmode(GPIO.BCM)
pin=21
pin2=20
GPIO.setup(pin, GPIO.OUT)
GPIO.setup(pin2, GPIO.OUT)
p=GPIO.PWM(pin,50)    #21号pin,50Hz
p2=GPIO.PWM(pin2,50)    #20号pin,50Hz
p.start(0)
p2.start(0)
# Set up camera constants
IM_WIDTH = 420
IM_HEIGHT = 420
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate
##################full??????############################
inpin1=13
inpin2=19
inpin3=5
inpin4=6
GPIO.setup(inpin1, GPIO.IN)
GPIO.setup(inpin2, GPIO.IN)
GPIO.setup(inpin3, GPIO.IN)
GPIO.setup(inpin4, GPIO.IN)
#############The flag#################
classflag=0
putflag=0
num1=0
sum1=0
sum2=0
sum3=0
sum4=0
sn=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
############video######################

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 45

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.
### USB webcam ###
#if camera_type == 'usb':
    # Initialize USB webcam feed
camera = cv2.VideoCapture(0)
ret = camera.set(3,IM_WIDTH)
ret = camera.set(4,IM_HEIGHT)
path1= '/home/pi/Desktop/get_img/test1.jpg'
img1 = cv2.imread(path1)
cv2.namedWindow('垃圾桶状态', 1)
cv2.resizeWindow('垃圾桶状态', (800,50))
cv2.moveWindow("垃圾桶状态", 100, 0)
p2.ChangeDutyCycle(8.4)
p.ChangeDutyCycle(8.9)
time.sleep(3)
p2.ChangeDutyCycle(0)
p.ChangeDutyCycle(0)
VIDEO_PATH = Path("/home/pi/tensorflow1/models/research/object_detection/test3.avi")#加粗的文字请自行替换成自己的路径跟文件名
player = OMXPlayer(VIDEO_PATH)
sleep(4)
player.quit()
### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        if(GPIO.input(inpin1)==0):
            img1 = cv2.imread(path1)
            cv2.putText(img1,"yes!!",(150,150),font,0.7,(0,0,255),1,cv2.LINE_AA)
        else:
            cv2.putText(img1,"no!!",(150,150),font,0.7,(0,0,255),1,cv2.LINE_AA)
        if(GPIO.input(inpin2)==0):
            img1 = cv2.imread(path1)
            cv2.putText(img1,"yes!!",(375,150),font,0.7,(0,0,255),1,cv2.LINE_AA)
        else:
            cv2.putText(img1,"no!!",(375,150),font,0.7,(0,0,255),1,cv2.LINE_AA)
        if(GPIO.input(inpin3)==0):
            img1 = cv2.imread(path1)
            cv2.putText(img1,"yes!!",(575,150),font,0.7,(0,0,255),1,cv2.LINE_AA)
        else:
            cv2.putText(img1,"no!!",(575,150),font,0.7,(0,0,255),1,cv2.LINE_AA)
        if(GPIO.input(inpin4)==0):
            img1 = cv2.imread(path1)
            cv2.putText(img1,"yes!!",(755,150),font,0.7,(0,0,255),1,cv2.LINE_AA)
        else:
            cv2.putText(img1,"no!!",(755,150),font,0.7,(0,0,255),1,cv2.LINE_AA)
        cv2.imshow('垃圾桶状态', img1)
        cv2.putText(img1," {0:.1f} ".format(sum1),(150,100),font,0.7,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(img1," {0:.1f} ".format(sum2),(370,100),font,0.7,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(img1," {0:.1f} ".format(sum3),(575,100),font,0.7,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(img1," {0:.1f} ".format(sum4),(765,100),font,0.7,(0,0,255),1,cv2.LINE_AA)
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)
        if((count1<=9)):
            if(int(classes[0][0])!=1):
                count1=0
        if((count2<=9)):
            if(int(classes[0][0])!=2):
                count2=0
        if((count3<=9)):
            if(int(classes[0][0])!=3):
                count3=0
        if((count4<=9)):
            if(int(classes[0][0])!=4):
                count4=0
        if((count5<=9)):
            if(int(classes[0][0])!=5):
                count5=0
        if((count6<=9)):
            if(int(classes[0][0])!=6):
                count6=0
        if((count7<=9)):
            if(int(classes[0][0])!=7):
                count7=0
        if((int(classes[0][0])==1)&(classflag==0)):
            count1+=1
            if(count1==10):
                classflag=1
                img1 = cv2.imread(path1)
                putflag=1
                if(int(classes[0][1])==1):
                    num1=2
                    sum2+=2
                elif(int(classes[0][1])!=1):
                    num1=1
                    sum2+=1
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(4.5)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(8.4)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
        elif((int(classes[0][0])==2)&(classflag==0)):
            count2+=1
            if(count2==10):
                classflag=2
                img1 = cv2.imread(path1)
                putflag=1
                if(int(classes[0][1])==2):
                    num1=2
                    sum3+=2
                elif(int(classes[0][1])!=2):
                    num1=1
                    sum3+=1
                p.ChangeDutyCycle(0)
                p.ChangeDutyCycle(5.6)
                time.sleep(2)
                p.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(4.5)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(8.4)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p.ChangeDutyCycle(8.9)
                time.sleep(2)
                p.ChangeDutyCycle(0)
        elif((int(classes[0][0])==3)&(classflag==0)):
            count3+=1
            if(count3==10):
                classflag=3
                img1 = cv2.imread(path1)
                putflag=1
                if(int(classes[0][1])==3):
                    num1=2
                    sum1+=2
                elif(int(classes[0][1])!=3):
                    num1=1
                    sum1+=1
                p.ChangeDutyCycle(0)
                p.ChangeDutyCycle(2.3)
                time.sleep(2)
                p.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(4.5)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(8.4)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p.ChangeDutyCycle(0)
                p.ChangeDutyCycle(8.9)
                time.sleep(2)
                p.ChangeDutyCycle(0)
        elif((int(classes[0][0])==4)&(classflag==0)):
            count4+=1
            if(count4==10):
                classflag=4
                img1 = cv2.imread(path1)
                putflag=1
                if(int(classes[0][1])==4):
                    num1=2
                    sum3+=2      
                elif(int(classes[0][1])!=4):
                    num1=1
                    sum3+=1
                p.ChangeDutyCycle(0)
                p.ChangeDutyCycle(5.6)
                time.sleep(2)
                p.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(4.5)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(8.4)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p.ChangeDutyCycle(0)
                p.ChangeDutyCycle(8.9)
                time.sleep(2)
                p.ChangeDutyCycle(0)
        elif((int(classes[0][0])==5)&(classflag==0)):
            count5+=1
            if(count5==10):
                classflag=5
                img1 = cv2.imread(path1)
                putflag=1
                if(int(classes[0][1])==5):
                    num1=2
                    sum4+=2
                elif(int(classes[0][1])!=5):
                    num1=1
                    sum4+=1
                p.ChangeDutyCycle(0)
                p.ChangeDutyCycle(12.3)
                time.sleep(2)
                p.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(4.5)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(8.4)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p.ChangeDutyCycle(0)
                p.ChangeDutyCycle(8.9)
                time.sleep(2)
                p.ChangeDutyCycle(0)
        elif((int(classes[0][0])==6)&(classflag==0)):
            count6+=1
            if(count6==10):
                classflag=6
                img1 = cv2.imread(path1)
                putflag=1
                if(int(classes[0][1])==6):
                    num1=2
                    sum4+=2
                elif(int(classes[0][1])!=6):
                    num1=1
                    sum4+=1
                p.ChangeDutyCycle(0)
                p.ChangeDutyCycle(12.3)
                time.sleep(2)
                p.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(4.5)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(8.4)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p.ChangeDutyCycle(0)
                p.ChangeDutyCycle(8.9)
                time.sleep(2)
                p.ChangeDutyCycle(0)
        elif((int(classes[0][0])==7)&(classflag==0)):
            count7+=1
            if(count7==10):
                classflag=7
                img1 = cv2.imread(path1)
                putflag=1
                if(int(classes[0][1])==7):
                    num1=2
                    sum2+=2
                elif(int(classes[0][1])!=7):
                    num1=1
                    sum2+=1
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(4.5)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p2.ChangeDutyCycle(8.4)
                time.sleep(2)
                p2.ChangeDutyCycle(0)
                p.ChangeDutyCycle(0)
        elif(int(classes[0][0])==8):
            count8+=1     
            if(count8==10):
             
                img1 = cv2.imread(path1)
                count1=0
                count2=0
                count3=0
                count4=0
                count5=0
                count6=0
                count7=0
                count8=0
                cv2.putText(img1,"nothing",(250,50),font,0.8,(0,0,255),0,cv2.LINE_AA)
                classflag=0
        if(putflag==1):
            sn+=1
            cv2.putText(img1,"ok!",(750,50),font,0.8,(0,0,255),0,cv2.LINE_AA)  
            putflag=0
        if(classflag==1):
            cv2.putText(img1," {0:.1f} ".format(num1),(650,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1," {0:.1f} ".format(sn),(180,50),font,1,(0,0,255),2,cv2.LINE_AA) 
            cv2.putText(img1,"Recyclable garbage cans",(250,50),font,0.8,(0,0,255),0,cv2.LINE_AA)
        if(classflag==2):
            cv2.putText(img1," {0:.1f} ".format(num1),(650,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"{0:.1f} ".format(sn),(180,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"Kitchen waste fruit",(250,50),font,0.8,(0,0,255),0,cv2.LINE_AA)
        if(classflag==3):
            cv2.putText(img1," {0:.1f} ".format(num1),(650,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"{0:.1f} ".format(sn),(180,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"Hazardous waste The battery",(250,50),font,0.8,(0,0,255),0,cv2.LINE_AA)
        if(classflag==4):
            cv2.putText(img1," {0:.1f} ".format(num1),(650,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"{0:.1f} ".format(sn),(180,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"Kitchen waste vegetables",(250,50),font,0.8,(0,0,255),0,cv2.LINE_AA)
        if(classflag==5):
            cv2.putText(img1," {0:.1f} ".format(num1),(650,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"{0:.1f} ".format(sn),(180,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"Other rubbish Cigarette butts",(250,50),font,0.8,(0,0,255),0,cv2.LINE_AA)
        if(classflag==6):
            cv2.putText(img1," {0:.1f} ".format(num1),(650,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"{0:.1f} ".format(sn),(180,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"Other rubbish Brick and tile",(250,50),font,0.8,(0,0,255),0,cv2.LINE_AA)
        if(classflag==7):
            cv2.putText(img1," {0:.1f} ".format(num1),(650,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"{0:.1f} ".format(sn),(180,50),font,1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img1,"Recyclable garbage Water bottles",(250,50),font,0.8,(0,0,255),0,cv2.LINE_AA)
        print(classes[0][0])
        # All the results have been drawn on the frame, so it's time to display it.          
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        #print('Object detector')

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()
cv2.destroyAllWindows()

