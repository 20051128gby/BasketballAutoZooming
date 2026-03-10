import json                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            #by gby
import re
import cv2
import numpy as np
from utils.plots import plot_one_box
from detector import Get_Pred
from PIL import Image
import sys
import cv2
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
from playsound import playsound
import tempfile
from collections import deque
import os
import time

x_center=0
y_center=0
import numpy as np
import myUtils
ball_model = YOLO(r"E:\vscodeP\AI\basketballAutoZooming\base_on_2021_Yolov5\basketballModel.pt")
final_list=[]
#pred_path = r'E:\vscodeP\data\Video\TestData.mp4'
pred_path = r"C:\test.mp4"
model_path = r'E:\vscodeP\AI\basketballAutoZooming\base_on_2021_Yolov5\weights\yolov5s.pt'

player_list=[]
player_centerList=[]
capture = cv2.VideoCapture(pred_path)
start_frame = 0 

# 跳转到指定帧
is_success = capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
if not is_success:
    print("无法跳转到指定帧。")

className=[]
confidenceScore=[]
referee=[]
# ret, img = capture.read()
ret, img = capture.read()
frame_height, frame_width, channels = img.shape 
video_filename = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 20
video = cv2.VideoWriter(video_filename, fourcc, fps, (1000, 600),True)
if(video.isOpened):
    print("isopen")






def detect_video():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           #by gby
    global capture
    global start_frame
    global confidenceScore
    global className
    global player_centerList
    cntFram = start_frame
    temp = 0
    prev_x, prev_y = 0, 0
    while True:
        ret, img = capture.read()
        temp+=1
        cntFram+=1
        if ret is not True:
            break
        detect = Get_Pred(model_path)

        ball_results_list = ball_model(img, verbose=False, conf=0.4)

        
        value_json = detect.process(img)
        value = json.loads(value_json)
        
        pred_xyxy = value['xyxy']
        cropList=myUtils.crop(img,pred_xyxy)

        final_list= myUtils.resize_with_white_background(cropList)        
        className,confidenceScore=myUtils.print_confidence_score(final_list, cntFram)
        #     for img in cropList:
        #         temp1+=1
        #         cv2.imwrite(str(temp1)+"_"+str(cntFram)+".jpg",img)
        #     temp=0

        # for i in range(len(className)):
        #     print(i)
        #     if(className[i]==1 and confidenceScore[i]>0.95):
        #         referee.append(cropList[i])
        player_list = []
        for i in range(len(className)):
            #print(i)
            if(className[i]==1 and confidenceScore[i]>0.95):
                player_list.append(pred_xyxy[i])
                plot_one_box(pred_xyxy[i], img, label=None, color=(255, 255, 0), line_thickness=1)
                            



        global x_center,y_center
            
            
        #             file_name =str(i)+"_"+str(cntFram) +".jpg"    
                  
        #             cv2.imwrite(file_name,np.array(cropList[i]))
        
        # for results in ball_results_list:
            
        #     for bbox in results.boxes.xyxy:
        #         x1, y1, x2, y2 = bbox[:4]

        #         x_center = (x1 + x2) / 2
        #         y_center = (y1 + y2) / 2
        #         # print(
        #         #     f"Ball coordinates: (x={x_center:.2f}, y={y_center:.2f})"
                
        #         # )
        #         #plot_one_box(bbox, img, label=None, color=(255, 255, 0), line_thickness=1)
        
    
        player_centerList = []
        for results in player_list:
            x1, y1, x2, y2 = results[:4]
            px_center = (x1 + x2) / 2
            py_center = (y1 + y2) / 2
            player_centerList.append((px_center,py_center))

        #x1,y1,x2,y2=myUtils.zoom1(frame_width,frame_height,player_centerList,(x_center,y_center),16,16,4,2)
        if(cntFram ==1):
            x,y = myUtils.zoom2(player_centerList)
        if(cntFram%10==0):
            prev_x, prev_y = myUtils.zoom2(player_centerList)
            #prev_x, prev_y = x, y
        if(x<prev_x-200):x+=10
        if(x>prev_x+200):x-=10
        if(y<prev_y-200):y+=10
        if(y>prev_y+200):y-=10

        if (x == 0 and y == 0) or (x-500<0 or x+500 > frame_width or y-300<0 or y+300 > frame_height):
            x, y = prev_x, prev_y
        
        # 更新上一次的x和y值
        


        #plot_one_box((x1,y1,x2,y2), img, label=None, color=(255, 255, 0), line_thickness=1)
        plot_one_box((x-500,y-300,x+500,y+300), img, label=None, color=(255, 255, 0), line_thickness=1)
        #print(x1,y1,x2,y2)
        #cv2.namedWindow("test",0)
        

        video.write(img[y-300:y+300, x-500:x+500])
        #cv2.imshow("test",img[y-300:y+300, x-500:x+500])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break




if __name__ == '__main__':
    detect_video()
    video.release()
    capture.release()
    cv2.destroyAllWindows()
    