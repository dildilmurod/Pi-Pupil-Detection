# show both camera streams 

import cv2
import numpy as np
import random as rng
import datetime
import os


class data_collector:
    landmarks = [(0.1, 0.1), (0.1, 0.9), (0.5, 0.5), (0.9, 0.9), (0.9, 0.1)]
    length = 20

    def __init__(self, window):
        self.black_window = window
        self.window_width = window.shape[0]
        self.window_height = window.shape[1]
        pass

    def add_cross(self):
        """ Add cross to the frame on specified landmarks """
        for i, j in data_collector.landmarks:
            cv2.line(self.black_window, (round(i * self.window_height) + data_collector.length,
                                        round(j * self.window_width) + data_collector.length),
                    (round(i * self.window_height) - data_collector.length,
                     round(j * self.window_width) - data_collector.length), color=(0, 0, 255), thickness=5)

            cv2.line(self.black_window, (round(i * self.window_height) - data_collector.length,
                                        round(j * self.window_width) + data_collector.length),
                    (round(i * self.window_height) + data_collector.length,
                     round(j * self.window_width) - data_collector.length), color=(0, 0, 255), thickness=5)
        return self.black_window

def create_workspace(path):
    timestm = datetime.datetime.now()
    time_right_now = timestm.strftime("%d_%m_%Y_%H_%M_%S")
    name_workspace = path + "Fixed_Marker_Capture" + time_right_now + '/'
    if not os.path.exists(name_workspace):
	    os.makedirs(name_workspace)
	    print("folder name", name_workspace)
    return name_workspace
	
srcPiCam = "1.avi"
#srcPiCam = 'libcamerasrc ! video/x-raw,width=640,height=480 ! videoflip method=clockwise ! videoconvert ! appsink drop=True'
pcap = cv2.VideoCapture(srcPiCam)


path = "/home/demo/Desktop/dataset/"
wkspacename = create_workspace(path)
print(wkspacename)
filename = wkspacename+'fixed.avi'

width= int(pcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(pcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#writer= cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
 

 
if pcap.isOpened():
        print(f'Pupil camera available:')
        

frame_count = 0

while True:
	frame_count = frame_count+1 
	pret, pframe = pcap.read()
	if pret: 
	    cv2.imshow('frame', pframe)
	    window_name = "Data_Collector"
	    window_size = [1920, 1080]
	    cv2.namedWindow(window_name)
	    cv2.resizeWindow(window_name, window_size[1], window_size[0])
	    drawing = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

	    experiment = data_collector(window=drawing)
	    test = experiment.add_cross()
	    cv2.imshow(window_name, test)
	    
	    #writer.write(pframe)
	    
	    
	if cv2.waitKey(1) & 0xFF == ord('q') or frame_count>=2000:
		break

pcap.release()
cv2.destroyAllWindows()
