import socket
import robot_actions as robot
import time
import os
import numpy as np
import cv2
from PIL import Image
import thread
from multiprocessing import Value,Queue
from naoqi import ALProxy
import sys

robotIp = "192.168.1.210"  # The IP and port address on which 'robot_actions.py' is listenting
port=9559
if len(sys.argv) > 1 and sys.argv[1]:
	robotIp=str(sys.argv[1])

t_step=20
camProxy = ALProxy("ALVideoDevice", robotIp, port)
resolution = 1    # VGA
colorSpace = 0   # Y channel

upper_cam = camProxy.subscribeCamera("Ucam",0, resolution, colorSpace, 5)
depth = camProxy.subscribeCamera("Dcam",2, resolution, colorSpace, 5)

basic_awareness = ALProxy("ALBasicAwareness",robotIp, port)

basic_awareness.setStimulusDetectionEnabled("People",True)
basic_awareness.setStimulusDetectionEnabled("Movement",True)
basic_awareness.setStimulusDetectionEnabled("Sound",True)
basic_awareness.setStimulusDetectionEnabled("Touch",True)

basic_awareness.setParameter("LookStimulusSpeed",0.7)
basic_awareness.setParameter("LookBackSpeed",0.5)
basic_awareness.setEngagementMode("FullyEngaged")
basic_awareness.setTrackingMode("Head")

tracker = ALProxy("ALTracker", robotIp, port)
targetName = "Face"
faceWidth = 0.1
tracker.registerTarget(targetName, faceWidth)


def cam(step,num):
	"""
	First get an image from Nao, then show it on the screen with PIL.
	"""
	e=open('files/episode.txt','rb')
	ep=e.read()
	ep = str(int(ep))
	e.close()

	save_path1='dataset/RGB/ep'+str(ep)+'/'
	save_path2='dataset/Depth/ep'+str(ep)+'/'
	if not os.path.exists(save_path1):
		os.makedirs(save_path1)
	if not os.path.exists(save_path2):
		os.makedirs(save_path2)

	# save_path1='dataset/RGB/ep'+str(ep[2])+str(ep[3])+'/'
	# save_path2='dataset/Depth/ep'+str(ep[2])+str(ep[3])+'/'
	t0 = time.time()

	# Get a camera image.
	# image[6] contains the image data passed as an array of ASCII chars.
	for i in range(1,9):
		yimg = camProxy.getImageRemote(upper_cam)
		dimg = camProxy.getImageRemote(upper_cam)#no depth cam in nao
		image=np.zeros((dimg[1], dimg[0]),np.uint8)
		values=map(ord,list(dimg[6]))
		j=0
		for y in range (0,dimg[1]):
			for x in range (0,dimg[0]):
				image.itemset((y,x),values[j])
				j=j+1
		name="depth_"+str(step)+"_"+str(i)+".png"
		complete_depth=os.path.join(save_path2,name)
		cv2.imwrite(complete_depth,image)
		im = Image.frombytes("L", (yimg[0], yimg[1]), yimg[6])
		name="image_"+str(step)+"_"+str(i)+".png"
		complete_rgb=os.path.join(save_path1,name)
		im.save(complete_rgb, "PNG")

	t1 = time.time()

	# Time the image transfer.

	print "acquisition delay ", t1 - t0
	num.value=1

def main():
	host='localhost'   # The IP address on which 'robot_listen.py' is listenting- It connects MDQN_k/data_generation_main.py to robot_listen.py
	port=12375
	s2=socket.socket()
	s2.bind((host,port))
	s2.listen(5)                 # Now wait for client connection.
	data = 'x'
	while(str(data)!='9'):  #wait start signal '9'
		c, addr = s2.accept()
		data = c.recv(1024);
	cam(1,Value('d', 0.0))
	c.send('9')
	for step in range(2,t_step+2):
		c, addr = s2.accept()
		data = c.recv(1024);
		print time.strftime('%Y/%m/%d-%H:%M:%S',time.localtime(time.time()))," recieve action ",str(data)

		num2= Value('d', 0.0)
		r=str(0)
		if str(data)=='1':

			thread.start_new_thread(cam,(step,num2,))
			r=robot.main(robotIp,data,step)

		else:
			basic_awareness.startAwareness()
			tracker.track(targetName)
			thread.start_new_thread(cam,(step,num2,))
			r=robot.main(robotIp,data,step)
			basic_awareness.stopAwareness()
			tracker.stopTracker()
		while num2.value==0:
			pass
		r=r+'\n'
		print time.strftime('%Y/%m/%d-%H:%M:%S',time.localtime(time.time()))," send reward ",r
		c.send(r)

	camProxy.unsubscribe(upper_cam)
	camProxy.unsubscribe(depth)


if __name__ == '__main__':

	main()
