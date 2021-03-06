import cv2.cv as cv
import cv2
from datetime import datetime
import time
import os
from multiprocessing import Pool
from tqdm import tqdm
import sys
import skvideo.io
import traceback
import pdb

PATH = "out/toCheck/"

class MotionDetectorInstantaneous():
    
    def onChange(self, val): #callback when the user change the detection threshold
        self.threshold = val
    
    def conv(self, img):
        bitmap = cv.CreateImageHeader((img.shape[1], img.shape[0]), cv.IPL_DEPTH_8U, 3)
        cv.SetData(bitmap, img.tostring(), img.dtype.itemsize * 3 * img.shape[1])
        return bitmap

    def getFrame(self):
        ret = False
        counter = 0
        while ret == False:
            ret, self.frame = self.capture.read()
            if counter > 100:
                return False
            counter += 1
        return True

    def __init__(self, fileName, threshold=1, showWindows=True, numFrameCheck=20):
        self.file = fileName
        self.frame = None
        self.capture=skvideo.io.VideoCapture(self.file)
        self.numFrameCheck = numFrameCheck
        ret = False
        print("A")
        self.getFrame()
        print("B")
        self.frame = self.conv(self.frame)
        self.res = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U)
        self.frame2gray = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t
        self.width = self.frame.width
        self.height = self.frame.height
        self.nb_pixels = self.width * self.height
        self.threshold = threshold

        self.frame = self.conv(self.frame)
        self.processImage(self.frame) #Process the image
        print("D")
        #Will hold the thresholded result
    def __call__(self):
        '''
        Returns true if motion is detected
        '''
        started = time.time()
        counter = 0
        thresh = 7
        skipFrames = int(cv2.VideoCapture(self.file).get(cv.CV_CAP_PROP_FRAME_COUNT)/self.numFrameCheck)
        for i in tqdm(range(self.numFrameCheck)):
            ret, curframe = self.capture.read()
            for i in range(skipFrames-1):
                ret, curframe = self.capture.read()
            if not self.getFrame():
                break
            curframe = self.conv(curframe)
            instant = time.time() #Get timestamp o the frame
            self.processImage(curframe) #Process the image
            if self.somethingHasMoved():
                counter += 1
            if counter >= thresh:
                return True
                
            cv.Copy(self.frame2gray, self.frame1gray)
            c=cv.WaitKey(1) % 0x100
            if c==27 or c == 10: #Break if user enters 'Esc'.
                break            
        return False
    

    def processImage(self, frame):
        cv.CvtColor(frame, self.frame2gray, cv.CV_RGB2GRAY)
        
        #Absdiff to get the difference between to the frames
        cv.AbsDiff(self.frame1gray, self.frame2gray, self.res)
        
        #Remove the noise and do the threshold
        cv.Smooth(self.res, self.res, cv.CV_BLUR, 5,5)
        cv.MorphologyEx(self.res, self.res, None, None, cv.CV_MOP_OPEN)
        cv.MorphologyEx(self.res, self.res, None, None, cv.CV_MOP_CLOSE)
        cv.Threshold(self.res, self.res, 10, 255, cv.CV_THRESH_BINARY_INV)

    def somethingHasMoved(self):
        nb=0 #Will hold the number of black pixels

        for x in range(self.height): #Iterate the hole image
            for y in range(self.width):
                if self.res[x,y] == 0.0: #If the pixel is black keep it
                    nb += 1
        avg = (nb*100.0)/self.nb_pixels #Calculate the average of black pixel in the image

        if avg > self.threshold:#If over the ceiling trigger the alarm
            return True
        else:
            return False

def fileCallback(fileAndMoved):
    file, itMoved = fileAndMoved
    if itMoved:
        print(file, "to", "out/Moving/"+file[file.rfind("/")+1:])
        os.rename(file, "out/Moving/"+file[file.rfind("/")+1:])
    else:
        print(file, "to", "out/NoMovementOrNoFaces/"+file[file.rfind("/")+1:])
        os.rename(file, "out/NoMovementOrNoFaces/"+file[file.rfind("/")+1:])

def wrapper(path):
    try:
        path = os.path.abspath(path)
        print("Checking", path)
        detect = MotionDetectorInstantaneous(path)
        return detect()
        # fileCallback(detect())
        print("Checked", path)
    except Exception, e:
        print("ERROR ON THREAD IN ISMOVING! filename:", path, "error:", e)
        traceback.format_exc()
        return False

if __name__=="__main__":
    detect = MotionDetectorInstantaneous("/Users/magenta/Desktop/YouTubeScraper/out/toCheck/3EjolpEEdcc.mp4")
    print(detect())
    # pool = Pool(processes=5)
    # results = []
    # for file in [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]:
    #     if ".mp4" in file:
    #         # fileCallback(detect())
    #         results = pool.apply_async(wrapper, args=(file, ))
    # pool.close()
    # pool.join()