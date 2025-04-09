import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import json
import math

import urllib.request

# telegram requirement dependencies
# set urlllib3 version to 2.2.3
import telegram.ext # version 13.14
from dotenv import load_dotenv # version 1.0.1
import os

import requests
        
# get token for telegram
load_dotenv()
Token = os.getenv("TOKEN")
updater = telegram.ext.Updater(Token, use_context = True)
dispatch = updater.dispatcher

model=YOLO('Experiment/Train3/TrainDVis_Train3_Best.pt')
# model = YOLO('./yolov8n.pt')
# model = YOLO('C:/Users/User/Downloads/experiment 22_2_2025/best (2).pt')
# Get class name
dict = model.names
class_list = list(dict.values())

cap = cv2.VideoCapture("./video/vid1_.mp4")

# get parking area annotation
with open('./streamlit/annotation/bounding_boxes_video_testing_new_1.json', 'r') as f:
    data = json.load(f)

parking_areas = []
for area in data:
    points = np.array(area["points"], np.int32)
    parking_areas.append(points)

while True:    
    ret,frame = cap.read()
    if not ret:
        break

    # frame=cv2.resize(frame,(1020,500))
    frame = cv2.resize(frame,(704,396))

    results=model.predict(frame)
    # print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    # print(px)
    
    my_list = []
    anomaly_list = []

    for index,row in px.iterrows():
#        print(row)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        if c.lower() == 'car':
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2

            for i, area in enumerate(parking_areas):
            
                results1 =cv2.pointPolygonTest(np.array(parking_areas[i],np.int32),((cx,cy)),False)
                if results1>=0:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                    conf = float(math.ceil(row[4]*100)/100) # math.ceil() digunakan untuk melakukan pembulatan keatas angka dibelakang koma
                    label_conf = f"{c}{conf}"
                    text_size = cv2.getTextSize(label_conf, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1+text_size[0] , y1-text_size[1]-3
                    cv2.rectangle(frame, (x1,y1), c2, [0,0,255], -1, cv2.LINE_AA)
                    cv2.putText(frame, label_conf, (x1,y1-2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
                    
                    # draw box
                    cv2.polylines(frame,[np.array(parking_areas[i],np.int32)],True,(255,0,0),2)
                    
                    my_list.append(c.lower())
        elif c.lower() != 'car':
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
                
            # Iterate through parking areas and check if the center is inside
            for i, area in enumerate(parking_areas):
            
                anomaly =cv2.pointPolygonTest(np.array(parking_areas[i],np.int32),((cx,cy)),False)
                if anomaly>=0:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                    conf = float(math.ceil(row[4]*100)/100) # math.ceil() digunakan untuk melakukan pembulatan keatas angka dibelakang koma
                    label_conf = f"Anomaly! {c}{conf}"
                    text_size = cv2.getTextSize(label_conf, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1+text_size[0] , y1-text_size[1]-3
                    cv2.rectangle(frame, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)
                    cv2.putText(frame, label_conf, (x1,y1-2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
                            
                    anomaly_list.append(c.lower())
    
    anmly = (len(anomaly_list))
    slot  = (len(parking_areas))
    o     = my_list.count('car')
    space = slot - o
    
    # show available parking slot
    cv2.putText(frame,str(space),(77,62),cv2.FONT_HERSHEY_PLAIN,5,(0,255,0),2)
    
    # give logic for telegram bot
    def start(update, context):
        update.message.reply_text("Hello! welcome to parking management system bot.")

    def helps(update, context):
        update.message.reply_text(
            """
            Hi there!, i'm telegram bot created by Reza. Please follow these commands :

            /start - to start the conversation
            /content - Information about parking slot
            /contact - Information about contact
            /help - to get this help menu

            i hope this helps :) 
            """
        )

    def content(update, context):
            if anmly < 1 : 
                update.message.reply_text(
                    f"Parkiran yang tersedia {space} dari {slot} yang tersedia"
                )
            else :
                update.message.reply_text(
                    f"""Parkiran yang tersedia {space} dari {slot} yang tersedia, Perhatian {anmly} Anomali terdeteksi!"""
                )

    def contact(update, context):
        update.message.reply_text(
            """
            Muhammad Reza Pahlevi (21360004)
            S1 Teknik Informatika
            Fakultas Sains Dan Teknologi Informasi
            Institut Sains Dan Teknologi Nasional
            Jakarta
            """
        )

    def handle_msg(update, context):
        update.message.reply_text(f"you said {update.message.text}")
    
    # initialize method
    dispatch.add_handler(telegram.ext.CommandHandler('start', start))
    dispatch.add_handler(telegram.ext.CommandHandler('help', helps))
    dispatch.add_handler(telegram.ext.CommandHandler('content', content))
    dispatch.add_handler(telegram.ext.CommandHandler('contact', contact))
    dispatch.add_handler(telegram.ext.MessageHandler(telegram.ext.Filters.text, handle_msg))

    updater.start_polling()
    
    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(0)& 0xFF==ord('q'):
        break

updater.stop()
cap.release()
cv2.destroyAllWindows()