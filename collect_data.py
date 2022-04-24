import cv2
import mediapipe as mp
import pyautogui as pag
import copy
#import tensorflow as tf
import os
from queue import Queue
from pymouse import PyMouseEvent
from threading import Thread
import numpy as np
import pandas as pd
#from model import MyModel

# m = MyModel()
# m.load_weights("./checkpoints/my_checkpoint")

grid_num = 25
window_width = 1920
window_height = 1080
grid_width = window_width/grid_num
grid_height = window_height/grid_num

# if os.path.exists("./dataset/faceGrid.csv"):
# 	df = pd.read_csv("./dataset/faceGrid.csv")

def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img


def set_ones(x,y,w,h):
    face = {'face_left_upcorner':[x,y],'face_right_downcorner':[x+w,y+h]}
    #find grid end
    for i in range(grid_num):
        for j in range(grid_num):
            grid_left_upcorner = [j*grid_width, i*grid_height]
            face_coordination = face['face_left_upcorner']
            if in_grid(grid_left_upcorner, face_coordination):
                start_coor = [j,i]
                break
        else:
            continue
        break
        
    #find grid end
    for i in range(grid_num,-1,-1):
        for j in range(grid_num,-1,-1):
            grid_left_upcorner = [j*grid_width, i*grid_height]
            face_coordination = face['face_right_downcorner']
            if in_grid(grid_left_upcorner, face_coordination):
                end_coor = [j,i]
                break
        else:
            continue
        break
        
    grid = np.zeros((grid_num,grid_num), dtype=int)
    for i in range(start_coor[1],end_coor[1]+1):
        for j in range(start_coor[0],end_coor[0]+1):
            grid[i,j] = 1
    #grid_df = pd.DataFrame(grid)
    #return grid
    return grid.flatten()
def in_grid(grid_left_upcorner,face_coordination):
    grid_right_downcorner = [grid_left_upcorner[0]+grid_width, grid_left_upcorner[1]+grid_width]
    if (grid_left_upcorner[0] <= face_coordination[0] < grid_right_downcorner[0]) and (grid_left_upcorner[1] <= face_coordination[1] < grid_right_downcorner[1]):
        return True
    else:
        return False



path = './dataset/test/'


X_SCALE = 0.001
Y_SCALE = 0.001


class clickEventListener(PyMouseEvent):
	def click(self, x, y, button, press):
		if button == 3:
			if press:
				x, y = pag.position()
				q.put((x, y))
		else:
			self.stop()


thread = Thread(target=clickEventListener().run)
thread.start()


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

q = Queue()

# 眼部点index
conn = [(263, 249),
		(249, 390),
		(390, 373),
		(373, 374),
		(374, 380),
		(380, 381),
		(381, 382),
		(382, 362),
		(263, 466),
		(466, 388),
		(388, 387),
		(387, 386),
		(386, 385),
		(385, 384),
		(384, 398),
		(398, 362),
		(33, 7),
		(7, 163),
		(163, 144),
		(144, 145),
		(145, 153),
		(153, 154),
		(154, 155),
		(155, 133),
		(33, 246),
		(246, 161),
		(161, 160),
		(160, 159),
		(159, 158),
		(158, 157),
		(157, 173),
		(173, 133)]
# 眼部点index集合
#con_set = {384, 385, 386, 387, 388, 133, 390, 263, 7, 398, 144, 145, 153, 154, 155, 157, 158, 159, 160, 33, 161, 163, 173, 466, 362, 373, 374, 246, 249, 380, 381, 382}
con_eyeLeft = {417, 282, 348, 446}
con_eyeRight = {226, 119, 193, 52}
con_Face = {234, 10, 454, 152}

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
cap.set(3,window_width)
cap.set(4,window_height)
with mp_face_mesh.FaceMesh(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5) as face_mesh:
	with open(path+"images_label.txt", "a", encoding="utf-8") as f:#, open('./dataset/faceGrid.txt','a') as Gridtxt:
		if os.path.exists(path+"appleFace"):
			image_i = len(os.listdir(path+"appleFace"))
		else:
			os.mkdir(path+"appleLeftEye")
			os.mkdir(path+"appleRightEye")
			os.mkdir(path+"appleFace")
			#Gridtxt = open('./dataset/faceGrid.txt','a')
			image_i = len(os.listdir(path+"appleFace"))

		if os.path.exists(path+"faceGrid.csv"):
			df = pd.read_csv(path+"faceGrid.csv")
			df.drop(df.columns[0], axis=1,inplace=True)

		while cap.isOpened():
			success, image = cap.read()
			if not success:
				print("Ignoring empty camera frame.")
				# If loading a video, use 'break' instead of 'continue'.
				continue
			image = cv2.resize(image, (int(image.shape[1]), int(image.shape[0])))
			image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
			image.flags.writeable = False
			results = face_mesh.process(image)
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			shape = image.shape

			#coordination initialization
			eyeLeft_x_max = eyeLeft_y_max = 0
			eyeLeft_x_min = eyeLeft_y_min = 99999

			eyeRight_x_max = eyeRight_y_max = 0
			eyeRight_x_min = eyeRight_y_min = 99999

			Face_x_max = Face_y_max = 0
			Face_x_min = Face_y_min = 99999


			origin_image = copy.deepcopy(image)
			if results.multi_face_landmarks:
				for face_landmarks in results.multi_face_landmarks:
					for index, landmark in enumerate(face_landmarks.landmark):
						# 获取眼部矩形坐标
						if index in con_eyeLeft:
							eyeLeft_x_max = max(eyeLeft_x_max, landmark.x)
							eyeLeft_x_min = min(eyeLeft_x_min, landmark.x)
							eyeLeft_y_max = max(eyeLeft_y_max, landmark.y)
							eyeLeft_y_min = min(eyeLeft_y_min, landmark.y)

						if index in con_eyeRight:
							eyeRight_x_max = max(eyeRight_x_max, landmark.x)
							eyeRight_x_min = min(eyeRight_x_min, landmark.x)
							eyeRight_y_max = max(eyeRight_y_max, landmark.y)
							eyeRight_y_min = min(eyeRight_y_min, landmark.y)

						if index in con_Face:
							Face_x_max = max(Face_x_max, landmark.x)
							Face_x_min = min(Face_x_min, landmark.x)
							Face_y_max = max(Face_y_max, landmark.y)
							Face_y_min = min(Face_y_min, landmark.y)

					mp_drawing.draw_landmarks(
						image=image,
						landmark_list=face_landmarks,
						connections=conn,
						landmark_drawing_spec=drawing_spec,
						connection_drawing_spec=drawing_spec)
			#eyeLeft
			eyeLeft_x_min, eyeLeft_y_max, eyeLeft_x_max, eyeLeft_y_min = int((1 - X_SCALE) * eyeLeft_x_min * shape[1]), int((Y_SCALE + 1) * eyeLeft_y_max * shape[0]), \
										 int((1 + X_SCALE) * eyeLeft_x_max * shape[1]), int((1 - Y_SCALE) * eyeLeft_y_min * shape[0])
			eyeLeft = origin_image[eyeLeft_y_min: eyeLeft_y_max, eyeLeft_x_min: eyeLeft_x_max]
			eyeLeft = cv2.resize(eyeLeft,(64,64))

			#eyeRight
			eyeRight_x_min, eyeRight_y_max, eyeRight_x_max, eyeRight_y_min = int((1 - X_SCALE) * eyeRight_x_min * shape[1]), int((Y_SCALE + 1) * eyeRight_y_max * shape[0]), \
										 int((1 + X_SCALE) * eyeRight_x_max * shape[1]), int((1 - Y_SCALE) * eyeRight_y_min * shape[0])
			eyeRight = origin_image[eyeRight_y_min: eyeRight_y_max, eyeRight_x_min: eyeRight_x_max]
			eyeRight = cv2.resize(eyeRight,(64,64))

			#Face
			Face_x_min, Face_y_max, Face_x_max, Face_y_min = int((1 - X_SCALE) * Face_x_min * shape[1]), int((Y_SCALE + 1) * Face_y_max * shape[0]), \
										 int((1 + X_SCALE) * Face_x_max * shape[1]), int((1 - Y_SCALE) * Face_y_min * shape[0])
			Face = origin_image[Face_y_min: Face_y_max, Face_x_min: Face_x_max]
			Face = cv2.resize(Face,(224,224))

			#cv2.rectangle(image, (Face_x_min,Face_y_min), (Face_x_max,Face_y_max), (0,0,255), 1, 8)

			# faceGrid = set_ones(Face_x_min, Face_y_min, Face_x_max - Face_x_min, Face_y_max - Face_y_min)
			# 画出眼睛观察的位置
			image[0:64, 0:64] = eyeRight
			image[0:64, 64:128] = eyeLeft

			image = draw_grid(image,(25,25))


			# out_win = "output_style_full_screen"
			# cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
			# cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
			# cv2.imshow(out_win, image)

			
			# cv2.putText(image, str(show_looking_pos[0]+','+str(show_looking_pos[1])), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

			cv2.namedWindow("resized",0);
			cv2.resizeWindow("resized", 1439, 899);
			cv2.imshow("resized",image)

			# cv2.resizeWindow('MediaPipe FaceMesh', 640, 640)
			# cv2.imshow('MediaPipe FaceMesh', image)

			#cv2.moveWindow('MediaPipe FaceMesh',400,0)
			# try:
			# 	cv2.imshow('eyes', eyeLeft)
			# except Exception as e:
			# 	print(e)
			if not q.empty():
				cv2.imwrite(f"{path}appleLeftEye/{image_i}.png", eyeLeft)
				cv2.imwrite(f"{path}appleRightEye/{image_i}.png", eyeRight)
				cv2.imwrite(f"{path}appleFace/{image_i}.png", Face)
				faceGrid = set_ones(Face_x_min, Face_y_min, Face_x_max - Face_x_min, Face_y_max - Face_y_min)
				#print(type(faceGrid))

				#Gridtxt = open('./dataset/faceGrid.txt','a')
				#Gridtxt.write(f"{image_i}\t{faceGrid}")

				if image_i ==0:
					df = pd.DataFrame([faceGrid])
				else:
					df_temp = pd.DataFrame([faceGrid])

					df.loc[image_i]=faceGrid
					#df = df.append(df_temp, ignore_index = True)
				# print(Face_x_min)
				# print(Face_x_max)
				# print(Face_y_min)
				# print(Face_y_max)

				looking_pos = q.get()

				print(f"{looking_pos[0]} {looking_pos[1]} {image_i}")
				f.write(f"{looking_pos[0]} {looking_pos[1]} {image_i}\n")

				image_i += 1
			if cv2.waitKey(5) & 0xFF == 27:
				break

df.to_csv(path+"faceGrid.csv",encoding = 'utf_8_sig')

cap.release()
thread.join()
