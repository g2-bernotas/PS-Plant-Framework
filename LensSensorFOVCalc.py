import math

def LensSensorFOVCalc(lens, sensorSize, camHeight):
    AOV_w =  math.degrees(2*math.atan((sensorSize[0]/2)/(lens)))
    AOV_h =  math.degrees(2*math.atan((sensorSize[1]/2)/(lens)))
    h=camHeight * math.tan(math.radians(AOV_w/2));
    v=camHeight * math.tan(math.radians(AOV_h/2)); 
    h=h*2; 
    v=v*2;
    
    return h,v
    
def LensSensorPixelCalc(lens, sensorSize, camHeight, resolution):
    h,v = LensSensorFOVCalc(lens, sensorSize, camHeight)
    h = h / resolution[0]
    v = v / resolution[1]
    return h,v