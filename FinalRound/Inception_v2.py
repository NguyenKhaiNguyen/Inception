import socket       
import sys      
import time
import cv2
import numpy as np
import json
import base64
import argparse
import logging
import os
import torch
import torch.nn.functional as F
from PIL import Image
from unet import UNet
import argparse
from scipy import stats
# from UNET import *
from ultralytics import YOLO
from unet import UNet
import math
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
port = 54321                
s.connect(('127.0.0.1', port)) 
traffic_sign = {"0":"cam_phai","1":"cam_trai","2":"car","3":"di_thang","4":"re_phai","5":"re_trai",}


angle = 0
speed = 500


pre_t = time.time()
error_arr = np.zeros(5)
def PID(error, p, i, d): #0.43,0,0.02
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    #print('DELAY: {:.6f}s'.format(delta_t))
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)



def sengment_long(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # image = cv2.resize(image, (320, 90))
    x = torch.from_numpy(image)
    x = x.to(device)
    x = x.transpose(1, 2).transpose(0, 1)
    x = x / 255.0
    x = x.unsqueeze(0).float()
    with torch.no_grad():
        # pretrainedUNET = weights(x)
        predictImage = net(x)
        predictImage = torch.sigmoid(predictImage)
        predictImage = predictImage[0]
        predictImage = predictImage.squeeze()
        predictImage = predictImage > 0.5
        predictImage = predictImage.cpu().numpy()
        predictImage = np.array(predictImage, dtype=np.uint8)
        predictImage = predictImage * 255
        predictImage = predictImage.transpose(1,2,0)
        
    return predictImage

def remove_small_contours(image):
    
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
   
    return image_remove


def morphology(b_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    b_img = cv2.morphologyEx(b_img, cv2.MORPH_OPEN, kernel, iterations=2)
    b_img = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=4)
    # b_img = cv2.dilate(b_img, kernel,iterations=3)
    b_img = remove_small_contours(b_img)
    # b_img = cv2.erode(b_img, kernel, iterations=3)
    return b_img

def control_speed(angle):
    if abs(angle) >= 0 and  abs(angle) <= 8:
        speed = 160
    elif abs(angle) > 8  and  abs(angle) <= 18:
        speed = 120
    elif abs(angle) > 18  and  abs(angle) <= 25:
        speed = 60
    else:
        speed = 10
    return speed


# def control_speed(angle):
#     if angle >= 0:
#         speed = -2.8*angle +160
#     else:
#         speed = 2.8*angle +160
#     return speed



def tracking_lane(center):
    
    center_point = (center, 270)
    center_point_original = (320,359)
    cv2.line(image, center_point, center_point_original,(0,255,0),2)

    error = center-320
    angle = PID(error, 1.3,0.0,0.1) #0.8,0.0,0.055
    speed = control_speed(angle)
    print(angle)
    print(speed)
    return angle, speed

def get_center(b_img, line):
    try:
        index_left = np.min(np.where(b_img[line,:] > 0))
        index_right = np.max(np.where(b_img[line,:] > 0))
    
        index_left = index_left*2
        index_right = index_right*2
        center = int((index_right - index_left)/2 + index_left)
        lane_width = index_right-index_left
    except:
        center = 320
        lane_width = 500
        index_left = 70
        index_right = 370
    # print("lane width: ", (index_right-index_left))
    return center, lane_width, index_left, index_right


def contour_area(b_img):
    contours = cv2.findContours(b_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for contour in contours:
        area = cv2.contourArea(contour)
    return area

def finding_line(b_img):
    canny = cv2.Canny(b_img, 100,200)
    try:
        lines = cv2.HoughLinesP(canny,1, np.pi/180, 100, minLineLength=150, maxLineGap=50)
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        goc_lech = np.arctan((y2-y1)/(x2-x1))
        if y1 >= y2:
            khoang_cach = int((y1-y2)/2 + y2)
        else:
            khoang_cach = int((y2-y1)/2 + y1)


        return lines, np.degrees(goc_lech), khoang_cach
    except:
        return None, None, None


def circle_area(boundingbox, tin_hieu):
    w = boundingbox.shape[1]
    h = boundingbox.shape[0]
    circle_list = []
    if tin_hieu == "re_phai" or tin_hieu == "re_trai" or tin_hieu == "di_thang":
        hsv = cv2.cvtColor(boundingbox, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,(80,120,0) ,(140,255,255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        hsv = cv2.cvtColor(boundingbox, cv2.COLOR_BGR2HSV)
        mask_1 = cv2.inRange(hsv,(150,50,0) ,(255,255,255))
        mask_2 = cv2.inRange(hsv,(0,50,0) ,(15,255,255))
        mask = cv2.bitwise_or(mask_1, mask_2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
    mask = cv2.Canny(mask, 100,200)
    if mask is not None:
        right = np.max(np.max(np.where(mask>0), axis=1))
        print(right)
        left = np.min(np.min(np.where(mask>0), axis=1))
        print(left)
        return (right - left)
    else:
        return None



control_mode = "tracking_lane"
count = 0
tin_hieu = ""
tiep_tuc_tim = False
if __name__ == "__main__":
    try:
        print("Loading Unet model......")
        # net = build_unet()
        net = UNet(n_channels=3, n_classes=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device=device)
        state_dict = torch.load("unet_v3_datagopduongnho.pth", map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        half = device.type != 'cpu'
        print("Loading YOLO model......")
        model = YOLO("yolov8m_v2.pt")
        print("Done. Start!!!")
        prev_frame_time = 0
        new_frame_time = 0
        message = bytes(f"{angle} {speed}", "utf-8")
        s.sendall(message)

        while True:
            # message = bytes(f"{angle} {speed}", "utf-8")
            # s.sendall(message)
            try:
                data = s.recv(1000000000)
            except:
                print("No received data")
                continue
           
            data_recv = json.loads(data)
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]

            # print("---------------------------------------")

            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)

#segment=============================================================================================
            frame = cv2.resize(image, (320,180))
            frame = frame[90:, :]
            b_img = sengment_long(frame)[:,:,1]
            b_img = morphology(b_img)
#detect==============================================================================================
            sign = []
            results = model(image, conf = 0.8)[0]
            count_dt = 0
            for box in results:
                if len(box) != 0:
                    class_bien_bao = int(box.boxes.cls)
                    conf = float(box.boxes.conf)
                    
                    # print(conf_1)
                    x_box = int(box.boxes.xywh[0,0])
                    y_box = int(box.boxes.xywh[0,1])
                    width_box = int(box.boxes.xywh[0,2])
                    heith_box = int(box.boxes.xywh[0,3])
                    cv2.rectangle(image, (int(x_box-width_box/2), int(y_box - heith_box/2)), (int(x_box+width_box/2), int(y_box + heith_box/2)), (0,255,0), 1)
                    cv2.putText(image, str(width_box), ((int(x_box), int(y_box + heith_box/2))), (cv2.FONT_HERSHEY_SIMPLEX), 1,(0,255,0), 2)


                    # # print(traffic_sign[str(class_bien_bao)])
                    if width_box >= 70:
                        tin_hieu = traffic_sign[str(class_bien_bao)]
                        print("Tin hieu:", traffic_sign[str(class_bien_bao)], "++++++++++")
                        count_dt += 1
                    #     sign.append([tin_hieu,conf])
                    if count_dt == 1:
                        break
                    # sign = np.array(sign)

                    # x_min = int(x_box-width_box/2)-5
                    # x_max = int(x_box+width_box/2)+5
                    # y_min = int(y_box - heith_box/2)-5
                    # y_max = int(y_box + heith_box/2)+5
               
                    # # print("width_box:",width_box)
                    # try:
                    #     cv2.imwrite("xla/bien_bao.png", image[y_min:y_max,x_min:x_max,:])
                    # except:
                    #     pass
                    # break
            # print("Tong sign: ", sign)

#control=============================================================================================
            
            print("__________________________________________________________________________")
            # message = bytes(f"{angle} {speed}", "utf-8")
            # s.sendall(message)
            if tin_hieu == "re_phai":
                _c, _lw, left_re, right_re = get_center(b_img, 50)
                if right_re >= 317:
                    angle = 0
                    speed = 90
                    message = bytes(f"{angle} {speed}", "utf-8")
                    s.sendall(message)
                    start_time = time.time()
                    print("--------------READY TO TURN RIGHT----------------")
                    while(time.time() - start_time < 1.3):
                        pass

                    data = s.recv(1000000000)
                    data_recv = json.loads(data)

                    angle = 25
                    speed = 90
                    message = bytes(f"{angle} {speed}", "utf-8")
                    s.sendall(message)
                    start_time = time.time()
                    print("--------------TURN RIGHT----------------")
                    while(time.time() - start_time < 1.1):
                        pass
                    tin_hieu = ""
            

                else:
                    center, lane_width,_,_ = get_center(b_img,65)
                    angle, speed = tracking_lane(center)
                    message = bytes(f"{angle} {speed}", "utf-8")
                    s.sendall(message) 

            elif tin_hieu == "re_trai":
                _c, _lw, left_re, right_re = get_center(b_img, 50)
                if left_re <= 3:
                    # angle = 0
                    # speed = 90
                    # message = bytes(f"{angle} {speed}", "utf-8")
                    # s.sendall(message)
                    # start_time = time.time()
                    # print("--------------READY TO TURN LEFT----------------")
                    # while(time.time() - start_time < 0.1):
                    #     continue

                    # data = s.recv(1000000000)
                    # data_recv = json.loads(data)

                    angle = -25
                    speed = 90
                    message = bytes(f"{angle} {speed}", "utf-8")
                    s.sendall(message)
                    start_time = time.time()
                    print("--------------TURN LEFT----------------")
                    while(time.time() - start_time < 1.1):
                        pass
                    
                    

                    tin_hieu = ""
               
                else:
                    center, lane_width,_,_ = get_center(b_img,65)
                    angle, speed = tracking_lane(center)
                    message = bytes(f"{angle} {speed}", "utf-8")
                    s.sendall(message) 



            elif tin_hieu == "di_thang":
             
                angle = 0
                speed = 150
                message = bytes(f"{angle} {speed}", "utf-8")
                s.sendall(message)
                start_time = time.time()
                print("--------------STRAIGHT----------------")
                while(time.time() - start_time < 0.2):
                    pass
                try:
                        data = s.recv(1000000000)
                except:
                    print("No received data")
                    continue
                        
                data_recv = json.loads(data)
                message = bytes(f"{angle} {speed}", "utf-8")
                s.sendall(message)
                tin_hieu = ""
               

            elif tin_hieu == "cam_phai":
                contour_trai = contour_area(b_img[:,:160])
                contour_phai = contour_area(b_img[:,160:])
                print("Trai:", contour_trai)
                print("Phai:", contour_phai)
                _c, _lw, left_re, right_re = get_center(b_img, 50)

                if right_re >= 317:
                    if abs(contour_trai-contour_phai) >= 3000:
                        cv2.imwrite("binary_img.png", b_img)
                        cv2.imwrite("frame_img.png", frame)
                        angle = 0
                        speed = 110
                        message = bytes(f"{angle} {speed}", "utf-8")
                        s.sendall(message)
                        start_time = time.time()
                        print("--------------STRAIGHT (NOT TURN RIGHT)----------------")
                        while(time.time() - start_time < 0.35):
                            pass
                        tin_hieu = ""
            
                    else:
                        angle = 0
                        speed = 150
                        message = bytes(f"{angle} {speed}", "utf-8")
                        s.sendall(message)
                        start_time = time.time()
                        print("--------------READY TO TURN LEFT (NOT TURN RIGHT)----------------")
                        while(time.time() - start_time < 0.9):
                            pass

                        data = s.recv(1000000000)
                        data_recv = json.loads(data)

                        angle = -25
                        speed = 90
                        message = bytes(f"{angle} {speed}", "utf-8")
                        s.sendall(message)
                        start_time = time.time()
                        print("--------------TURN LEFT (NOT TURN RIGHT)----------------")
                        while(time.time() - start_time < 1.3):
                            pass
                            
                        tin_hieu = ""
 
                else:
                    center, lane_width,_,_ = get_center(b_img,65)
                    angle, speed = tracking_lane(center)
                    message = bytes(f"{angle} {speed}", "utf-8")
                    s.sendall(message) 



            elif tin_hieu == "cam_trai":
                contour_trai = contour_area(b_img[:,:160])
                contour_phai = contour_area(b_img[:,160:])
                print("Trai:", contour_trai)
                print("Phai:", contour_phai)
                _c, _lw, left_re, right_re = get_center(b_img, 50)

                if left_re <= 10:
                    if abs(contour_trai-contour_phai) >= 3000:
                        cv2.imwrite("binary_img.png", b_img)
                        cv2.imwrite("frame_img.png", frame)
                        angle = 0
                        speed = 110
                        message = bytes(f"{angle} {speed}", "utf-8")
                        s.sendall(message)
                        start_time = time.time()
                        print("--------------STRAIGHT (NOT TURN LEFT)----------------")
                        while(time.time() - start_time < 0.35):
                            pass
                        tin_hieu = ""
            
                    else:
                        # angle = 0
                        # speed = 150
                        # message = bytes(f"{angle} {speed}", "utf-8")
                        # s.sendall(message)
                        # start_time = time.time()
                        # print("--------------READY TO TURN RIGHT (NOT TURN LEFT)----------------")
                        # while(time.time() - start_time < 0.1):
                        #     continue

                        # data = s.recv(1000000000)
                        # data_recv = json.loads(data)

                        angle = 25
                        speed = 90
                        message = bytes(f"{angle} {speed}", "utf-8")
                        s.sendall(message)
                        start_time = time.time()
                        print("--------------TURN RIGHT (NOT TURN LEFT)----------------")
                        while(time.time() - start_time < 1.3):
                            pass
                            
                        tin_hieu = ""
 
                else:
                    center, lane_width,_,_ = get_center(b_img,65 )
                    angle, speed = tracking_lane(center)
                    message = bytes(f"{angle} {speed}", "utf-8")
                    s.sendall(message) 



            elif tin_hieu == "car":
                if width_box >= 100:
                    if x_box < 320:
                        angle = 1
                        speed = 70
                    else: 
                        angle = -1
                        speed = 70

                    message = bytes(f"{angle} {speed}", "utf-8")
                    s.sendall(message)
                    start_time = time.time()
                    print("--------------CAR----------------")
                    while(time.time() - start_time < 0.4):
                        pass
                    tin_hieu = ""
                   
                else:
                    
                #     center, lane_width,_,_ = get_center(b_img,30)
                #     angle, speed = tracking_lane(center)
                #     message = bytes(f"{angle} {speed}", "utf-8")
                #     s.sendall(message)
                #     tin_hieu = ""
                # else:
                    center, lane_width,_,_ = get_center(b_img,65)
                    angle, speed = tracking_lane(center)
                    message = bytes(f"{angle} {speed}", "utf-8")
                    s.sendall(message)

            else:
                center, lane_width,_,_ = get_center(b_img,65)
                angle, speed = tracking_lane(center)
                message = bytes(f"{angle} {speed}", "utf-8")
                s.sendall(message) 



            # message = bytes(f"{angle} {speed}", "utf-8")
            # s.sendall(message) 
            # canny = cv2.Canny(b_img, 100,200)
            # lines = cv2.HoughLinesP(canny,1, np.pi/180, 100, minLineLength=120, maxLineGap=50)
            # if lines is not None:
            #     for line in lines:
            #         x1,y1,x2,y2 = line[0]
            #         cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
            
            b_img[65,:] = 1
            b_img[30,:] = 1
  
            cv2.imshow("Image", image)
            # cv2.imshow("Frame", frame)
            cv2.imshow("Binary", b_img)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
 
                cv2.imwrite("binary_img.png", b_img)
                cv2.imwrite("frame_img.png", frame)
                break
    finally:
        
        print('closing socket')
        s.close()




# elif tin_hieu == "re_trai":
            #     dien_tich_circle = 0
            #     try:
            #         radius = circle_area(image[y_min:y_max,x_min:x_max,:], tin_hieu)
            #         dien_tich_circle = radius*radius*np.pi
            #         print("Dien tich circle:", dien_tich_circle)
            #     except:
            #         pass
            #     if dien_tich_circle >= 16000:
            #         angle = 0
            #         speed = 110
            #         message = bytes(f"{angle} {speed}", "utf-8")
            #         s.sendall(message)
            #         start_time = time.time()
            #         while(time.time() - start_time < 0.75):
            #             print("^^^^^^^^^^^^^^^^^^^^^^^^^^")

            #         data = s.recv(1000000000)
            #         data_recv = json.loads(data)

            #         angle = -25
            #         speed = 90
            #         message = bytes(f"{angle} {speed}", "utf-8")
            #         s.sendall(message)
            #         start_time = time.time()
            #         while(time.time() - start_time < 1.1):
            #             print("<=<=<=<=<=<=<=<=<=<=<=<=<=<=")

            #         tin_hieu = ""
            #         width_box = 0

            #     else:
                    
            #         center, lane_width = get_center(b_img)
            #         angle, speed = tracking_lane(center)
            #         message = bytes(f"{angle} {speed}", "utf-8")
            #         s.sendall(message)

            # elif tin_hieu == "car":
            #     if width_box >= 75:
            #         if x_box < 320:
            #             angle = 3
            #             speed = 90
            #         else: 
            #             angle = -3
            #             speed = 90

            #         message = bytes(f"{angle} {speed}", "utf-8")
            #         s.sendall(message)
            #         start_time = time.time()
            #         while(time.time() - start_time < 0.1):
            #             print("--------------CAR----------------")
            #         tin_hieu = ""
            #         width_box = 0
            #     else:
                    
            #         center, lane_width = get_center(b_img)
            #         angle, speed = tracking_lane(center)
            #         message = bytes(f"{angle} {speed}", "utf-8")
            #         s.sendall(message)
                
            # elif tin_hieu == "di_thang":
            #     if width_box >= 75:
            #         angle = 0
            #         speed = 200
            #         message = bytes(f"{angle} {speed}", "utf-8")
            #         s.sendall(message)
            #         start_time = time.time()
            #         while(time.time() - start_time < 0.5):
            #             print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
            #         tin_hieu = ""
            #         width_box = 0
            #     else:
            #         center, lane_width = get_center(b_img)
            #         angle, speed = tracking_lane(center)
            #         message = bytes(f"{angle} {speed}", "utf-8")
            #         s.sendall(message)

            # elif tin_hieu == "cam_phai":
            #     contour_trai = contour_area(b_img[:,:160])
            #     contour_phai = contour_area(b_img[:,160:])
            #     print("Trai:", contour_trai)
            #     print("Phai:", contour_phai)
            #     dien_tich_circle = 0
            #     try:
            #         radius = circle_area(image[y_min:y_max,x_min:x_max,:], tin_hieu)
            #         dien_tich_circle = radius*radius*np.pi
            #         print("Dien tich circle:", dien_tich_circle)
            #     except:
            #         pass
            #     if dien_tich_circle > 20000:
            #         if abs(contour_trai-contour_phai) >= 2000:
            #             cv2.imwrite("binary_img.png", b_img)
            #             cv2.imwrite("frame_img.png", frame)
            #             angle = 0
            #             speed = 110
            #             message = bytes(f"{angle} {speed}", "utf-8")
            #             s.sendall(message)
            #             start_time = time.time()
            #             while(time.time() - start_time < 0.35):
            #                 print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
            #             tin_hieu = ""
            #             width_box = 0
            #         else:
            #             angle = 0
            #             speed = 90
            #             message = bytes(f"{angle} {speed}", "utf-8")
            #             s.sendall(message)
            #             start_time = time.time()
            #             while(time.time() - start_time < 0.3):
            #                 print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
            #                 print("##########################")

            #             data = s.recv(1000000000)
            #             data_recv = json.loads(data)

            #             angle = -25
            #             speed = 90
            #             message = bytes(f"{angle} {speed}", "utf-8")
            #             s.sendall(message)
            #             start_time = time.time()
            #             while(time.time() - start_time < 1.3):
            #                 print("<=<=<=<=<=<=<=<=<=<=<=<=<=<=")
            #                 print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

            #             tin_hieu = ""
            #             width_box = 0
            #     else:
            #         center, lane_width = get_center(b_img)
            #         angle, speed = tracking_lane(center)
            #         message = bytes(f"{angle} {speed}", "utf-8")
            #         s.sendall(message) 
                
            # elif tin_hieu == "cam_trai":
            #     contour_trai = contour_area(b_img[:,:160])
            #     contour_phai = contour_area(b_img[:,160:])
            #     print("Trai:", contour_trai)
            #     print("Phai:", contour_phai)
            #     dien_tich_circle = 0
            #     try:
            #         radius = circle_area(image[y_min:y_max,x_min:x_max,:], tin_hieu)
            #         dien_tich_circle = radius*radius*np.pi
            #         print("Dien tich circle:", dien_tich_circle)
            #     except:
            #         pass
            #     if dien_tich_circle > 15000:
            #         if abs(contour_trai-contour_phai) >= 7000:
            #             cv2.imwrite("binary_img.png", b_img)
            #             cv2.imwrite("frame_img.png", frame)
            #             angle = 0
            #             speed = 110
            #             message = bytes(f"{angle} {speed}", "utf-8")
            #             s.sendall(message)
            #             start_time = time.time()
            #             while(time.time() - start_time < 0.4):
            #                 print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
            #             tin_hieu = ""
            #             width_box = 0
            #         else:
            #             angle = 0
            #             speed = 90
            #             message = bytes(f"{angle} {speed}", "utf-8")
            #             s.sendall(message)
            #             start_time = time.time()
            #             while(time.time() - start_time < 0.45):
            #                 print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
            #                 print("##########################")

            #             data = s.recv(1000000000)
            #             data_recv = json.loads(data)

            #             angle = 25
            #             speed = 90
            #             message = bytes(f"{angle} {speed}", "utf-8")
            #             s.sendall(message)
            #             start_time = time.time()
            #             while(time.time() - start_time < 1.3):
            #                 print("=>=>=>=>=>=>=>=>=>=>=>=>=>")
            #                 print(">>>>>>>>>>>>>>>>>>>>>>>>>>")

            #             tin_hieu = ""
            #             width_box = 0

            #     else:
            #         center, lane_width = get_center(b_img)
            #         angle, speed = tracking_lane(center)
            #         message = bytes(f"{angle} {speed}", "utf-8")
            #         s.sendall(message) 

                