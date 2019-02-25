# Python wrapper for FlyCap2 api
import PyCapture2
from sys import exit
from time import sleep
import os
import serial
from timeit import default_timer as timer

def printBuildInfo():
	libVer = PyCapture2.getLibraryVersion()
	print("FlyCapture2 library version:", libVer[0], libVer[1], libVer[2], libVer[3])

def printCameraInfo(cam):
	camInfo = cam.getCameraInfo()
	print("\n*** CAMERA INFORMATION ***\n")
	print("Serial number -", camInfo.serialNumber)
	print("Camera model -", camInfo.modelName)
	print("Camera vendor -", camInfo.vendorName)
	print("Sensor -", camInfo.sensorInfo)
	print("Resolution -", camInfo.sensorResolution)
	print("Firmware version -", camInfo.firmwareVersion)
	print("Firmware build time -", camInfo.firmwareBuildTime)

def checkSoftwareTriggerPresence(cam):
	triggerInq = 0x530
	if(cam.readRegister(triggerInq) & 0x10000 != 0x10000):
		return False
	return True

def pollForTriggerReady(cam):
	softwareTrigger = 0x62C
	while True:
		regVal = cam.readRegister(softwareTrigger)
		if not regVal:
			break

def fireSoftwareTrigger(cam):
	softwareTrigger = 0x62C
	fireVal = 0x80000000
	cam.writeRegister(softwareTrigger, fireVal)

def outputCameraProperties(c):
		t=c.getProperty(PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE)
		frameRate = c.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE).absValue
		autoExp = c.getProperty(PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE).absValue
		shutter = c.getProperty(PyCapture2.PROPERTY_TYPE.SHUTTER).absValue
		brightness = c.getProperty(PyCapture2.PROPERTY_TYPE.BRIGHTNESS).absValue
		gain = c.getProperty(PyCapture2.PROPERTY_TYPE.GAIN).absValue
		wb = c.getProperty(PyCapture2.PROPERTY_TYPE.WHITE_BALANCE).absValue
		print('framerate: {}, autoexp: {}; shutter: {}, brightness: {}, gain: {}, wb: {}'.format(frameRate, autoExp, shutter, brightness, gain, wb))
		
def captureImages(comPort, toBeProcessedDir, mode):	
	printBuildInfo()
	bus = PyCapture2.BusManager()
	numCams = bus.getNumOfCameras()
	if not numCams:
		print("Insufficient number of cameras. Exiting...")
		return -1
		
	# Initiate Serial connection
	ser = serial.Serial(comPort, 9600, timeout=1) # 1s timeout
	sleep(1) 
	ser.reset_input_buffer()
	ser.reset_output_buffer()
	c = PyCapture2.Camera()
	c.connect(bus.getCameraFromIndex(0))
	
	#power on the Camera
	cameraPower = 0x610
	powerVal = 0x80000000
	
	c.writeRegister(cameraPower, powerVal)
	
	#waiting for Camera to power up
	retries = 10
	timeToSleep = 0.1	#seconds
	for i in range(retries):
		sleep(timeToSleep)
		try:
			regVal = c.readRegister(cameraPower)
		except PyCapture2.Fc2error:	
			pass
		awake = True
		if regVal == powerVal:
			break
		awake = False
	if not awake:
		print("Could not wake Camera. Exiting...")
		return -1
	printCameraInfo(c)
	
	triggerMode = c.getTriggerMode()
	triggerMode.onOff = True
	triggerMode.mode = 0
	triggerMode.parameter = 0
	triggerMode.source = 7		
	
	c.setTriggerMode(triggerMode)
	pollForTriggerReady(c)
	
	grabM=PyCapture2.GRAB_MODE.BUFFER_FRAMES
	c.setConfiguration(numBuffers=20, grabTimeout = 5000, grabMode=grabM) 
	
	c.startCapture()
	
	if not checkSoftwareTriggerPresence(c):	
		print("SOFT_ASYNC_TRIGGER not implemented on this Camera! Stopping application")
		return -1
	
	fireSoftwareTrigger(c)
	image =c.retrieveBuffer()
	pix=PyCapture2.PIXEL_FORMAT.RAW8
	imAmbient = image.convert(pix); 
	
	numImages = 4;
	if mode  == "VISNIR":
		numImages=8;
	
	offset = 4 if mode == "NIR"  else 0 
	imgs=[]
	start = timer()
	for i in range(numImages):
		pollForTriggerReady(c)
		serC=str(i+offset).encode()
		ser.write(serC)
		t=ser.read(1) 
		if not t.decode() == str(i+offset):
			print("Error: Invalid response from Serial port")
			return -1
			
		fireSoftwareTrigger(c)
		image = c.retrieveBuffer()
		im1=image.convert(pix) 
		imgs.append(im1)
		
	ser.write(b'9') # Switch off all the lights
	
	for i,image in enumerate(imgs):
		error=image.save(os.path.join(toBeProcessedDir, ("im" + str(i) + ".bmp")).encode("utf-8"), PyCapture2.IMAGE_FILE_FORMAT.BMP);
	
	error=imAmbient.save(os.path.join(toBeProcessedDir, "imAmbient.bmp").encode("utf-8"), PyCapture2.IMAGE_FILE_FORMAT.BMP);
	
	print('Time taken: {}'.format( timer() - start))
	
	c.setTriggerMode(onOff = False)
	print("Finished grabbing images!")
	c.stopCapture()
	c.disconnect()
		
	ser.close()
	return 0