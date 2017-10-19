import robot_actions as robot
from naoqi import ALProxy
from PIL import Image
import os
import random
import time
robotIp = '192.168.1.163'
port = 9559
actiong_dict={1:"wait",2:"look toward human",3:"hello",4:"shake hand"}

resolution = 1    # VGA
colorSpace = 0   # Y channel
camProxy = ALProxy("ALVideoDevice", robotIp, port)
upper_cam = camProxy.subscribeCamera("Ucam",0, resolution, colorSpace, 5)
ep ='x'
save_path1='dataset/RGB/ep'+str(ep)+'/'
save_path2='dataset/Depth/ep'+str(ep)+'/'
if not os.path.exists(save_path1):
	os.makedirs(save_path1)
def saveImg(step):
    for i in range(1,9):
		yimg = camProxy.getImageRemote(upper_cam)
		# dimg = camProxy.getImageRemote(upper_cam)#no depth cam in nao
		# image=np.zeros((dimg[1], dimg[0]),np.uint8)
		# values=map(ord,list(dimg[6]))
		# j=0
		# for y in range (0,dimg[1]):
		# 	for x in range (0,dimg[0]):
		# 		image.itemset((y,x),values[j])
		# 		j=j+1
		# name="depth_"+str(step)+"_"+str(i)+".png"
		# complete_depth=os.path.join(save_path2,name)
		# cv2.imwrite(complete_depth,image)
		im = Image.frombytes("L", (yimg[0], yimg[1]), yimg[6])
		name="image_"+str(step)+"_"+str(i)+".png"
		complete_rgb=os.path.join(save_path1,name)
		im.save(complete_rgb, "PNG")
def main():
    step=0
    while True:
        saveImg(step)
        action = random.randint(1,4)
        robot.main(robotIp,action,0)
        print time.strftime('%Y/%m/%d-%H:%M:%S',time.localtime(time.time()))," performing action ",actiong_dict[action]
        step+=1
        time.sleep(2)
if __name__ == '__main__':
    main()
