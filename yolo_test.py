#!/usr/bin/env python3

import rospy
import threading, requests, time
import math
import cv2
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped , Vector3
from mavros_msgs.msg import State , AttitudeTarget
from mavros_msgs.srv import SetMode, CommandBool , SetModeRequest , SetMavFrame , SetMavFrameRequest
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import NavSatFix , Image
from cv_bridge import CvBridge, CvBridgeError
import datetime
from ultralytics import YOLO
from yolo_segmentation import YOLOSegmentation



class PID:
    def __init__(self, kp, ki, kd, max_output, min_output, max_integ, min_integ, sample_time):
        self.kp = kp  # Proportional Gain
        self.ki = ki  # Integral Gain
        self.kd = kd  # Derivative Gain
        self.max_output = max_output  # Maximum Output
        self.min_output = min_output  # Minimum Output
        self.max_integ = max_integ  # Maximum Integral Term
        self.min_integ = min_integ  # Minimum Integral Term
        self.sample_time = sample_time  # Sample Time

        self.target = 0.0  # Target Value
        self.integ = 0.0  # Integral Term
        self.last_error = 0.0  # Last Error
        self.last_time = rospy.Time.now()  # Last Time

    def update(self, feedback_value):
        error = self.target - feedback_value  # Error
        dt = (rospy.Time.now() - self.last_time).to_sec()  # Time Step

        # Proportional Term
        P = self.kp * error

        # Integral Term
        self.integ += error * dt
        self.integ = max(self.min_integ, min(self.max_integ, self.integ))
        I = self.ki * self.integ

        # Derivative Term
        D = self.kd * (error - self.last_error) / dt

        # PID Output
        output = P + I + D
        output = max(self.min_output, min(self.max_output, output))

        # Update Last Error and Last Time
        self.last_error = error
        self.last_time = rospy.Time.now()

        return output

class MoveDrone:
    def __init__(self):
        rospy.init_node('move_drone', anonymous=True)


        # load the pre-trained YOLOv8n model
        self.ys = YOLOSegmentation("yolov8n-seg.pt")
        #self.model = YOLO("yolov8n-seg.pt")
           # CvBridge
        self.bridge = CvBridge()  

        # Initialize ROS Subscriber
        rospy.Subscriber('/mavros/state', State, self.state_cb)
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.position_cb)
        rospy.Subscriber("/mavros/global_position/global" , NavSatFix , self.callBaclkGlobalPosition)
        rospy.Subscriber("/standard_vtol/camera/rgb/image_raw", Image, self.image_callback)


        # Initialize ROS Publisher
        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
        self.alt_pub = rospy.Publisher('/mavros/setpoint_position/rel_alt', Float32, queue_size=10)
        self.att_pub = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)

       
        # Initialize ROS Service
        self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.flight_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)

          # frame변경
        self.set_mav_frame =rospy.ServiceProxy("mavros/setpoint_velocity/mav_frame", SetMavFrame)     


        # Initialize Variables
        self.current_state = State()
        self.current_pose = None
        self.current_global_position = None
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = "home"
        self.target_pose.pose.position.x = 10.0
        self.target_pose.pose.position.y = 10.0
        self.target_pose.pose.position.z = 0.0
        self.offb_set_mode = SetModeRequest()
        self.offb_set_mode.custom_mode = 'OFFBOARD'
        
        self.frame_set_mode = SetMavFrameRequest()
        self.frame_set_mode.mav_frame = 8

        self.att_msg = AttitudeTarget()
        self.att_msg.header.stamp = rospy.Time.now()
        self.att_msg.type_mask = 128  # 롤, 피치, 요륙 속도 제어



        # tracking 관련 target_lat = 35.8934302
        self.target_lat = None    
        self.target_lng = None     

     

        # Initialize PID Controllers
        self.x_pid = PID(0.2, 0.01, 0.01, 1.0, -1.0, 1.0, -1.0, 0.1)
        self.y_pid = PID(0.2, 0.01, 0.01, 1.0, -1.0, 1.0, -1.0, 0.1)

        self.target_center_x = None
        self.target_center_ｙ = None

        self.image_center_x = None
        self.image_center_y = None



        # define some constants
        self.CONFIDENCE_THRESHOLD = 0.8
        self.GREEN = (0, 255, 0)





    def show_image(self,img):
        cv2.imshow("Image Window", img)
        
        cv2.waitKey(1)

    # Define a callback for the Image message
    def image_callback(self,img_msg):
        # log some info about the image topic
        #rospy.loginfo(img_msg.header)

        # Try to convert the ROS Image message to a CV2 Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8") # color
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        bboxes, classes, segmentations, scores , flag= self.ys.detect(cv_image)

        if flag == 1:
            for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
                # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
                (x, y, x2, y2) = bbox
                # if class_id == 32:
                #     cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)

                #     cv2.polylines(img, [seg], True, (0, 0, 255), 4)

                #     cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


                cv2.rectangle(cv_image, (x, y), (x2, y2), (255, 0, 0), 1)

                cv2.polylines(cv_image, [seg], True, (0, 0, 255), 1)

                cv2.putText(cv_image, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # # run the YOLO model on the frame
        # detections = self.model(cv_image)[0]
        # # loop over the detections
        # for data in detections.boxes.data.tolist():
        #     # extract the confidence (i.e., probability) associated with the detection
        #     confidence = data[4]

        #     # filter out weak detections by ensuring the 
        #     # confidence is greater than the minimum confidence
        #     if float(confidence) < self.CONFIDENCE_THRESHOLD:
        #         continue

        #     # if the confidence is greater than the minimum confidence,
        #     # draw the bounding box on the frame
        #     xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        #     cv2.rectangle(cv_image, (xmin, ymin) , (xmax, ymax), self.GREEN, 2)
        self.show_image( cv_image)


    def callBaclkGlobalPosition(self,data):
        #print(data)     
        self.current_global_position = data

        #sub_topics_ready['global_pos'] = True


    def setOffboard(self):
        last_req = rospy.Time.now()
        
        while(not rospy.is_shutdown()):
            if(self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if((self.flight_mode_service.call(self.offb_set_mode).mode_sent == True)):
                    #self.flight_mode_service.call(self.offb_set_mode)
                    rospy.loginfo("OFFBOARD enabled")
                    break
            # else:
            #     break

    def state_cb(self, state_msg):
        self.current_state = state_msg

    def position_cb(self, odom_msg):
        self.current_pose = odom_msg.pose.pose

    def arm(self):
        rospy.loginfo("Arming Drone")       
        while not rospy.is_shutdown():
            if self.current_state.armed:
                break
            self.arm_service(True)
            rospy.sleep(1)
      

    def takeoff(self):
        rospy.loginfo("Taking off")
        while not rospy.is_shutdown():
            # print(self.current_state.mode)
            # if self.current_pose.position.z >= 3.0:
            #     break
            # self.alt_pub.publish(3.0)

            vel_msg = Twist()
            vel_msg.linear.x = 1500.0
            vel_msg.linear.y = 0.0
            vel_msg.linear.z = 2100.0
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.z = 0.0
            self.vel_pub.publish(vel_msg)
            rospy.sleep(0.1)

    def move_drone(self):
        rospy.loginfo("Moving Drone")

        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
    
            rate.sleep()

    # tracking 관련 함수
    def distance_on_xy(self, lat1, lon1, lat2, lon2):
        R = 6371000  # 지구의 반지름 (m)
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        dx = R * math.cos(lat1_rad) * (lon2_rad - lon1_rad)
        dy = R * (lat2_rad - lat1_rad)
        return dx, dy 

if __name__ == "__main__":

    move_drone = MoveDrone()

     # Arm Drone
    #move_drone.arm()

    # Takeoff Drone
    #move_drone.takeoff()
    #rospy.sleep(2.0)

    move_drone.move_drone()
