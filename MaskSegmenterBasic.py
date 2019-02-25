from skimage.filters import threshold_otsu
from separateIntoPlots import createPlantMask, cleanMask
import numpy as np

class MaskSegmenterBasic():
    def __init__(self, params):
        self.im = params['im'] 
        self.T = params['T'] 
        self.minSize = params['minSize']
        self.minImg = params['minImg']
        self.diskSize = params['diskSize']
    
    def segmentImage(self):
        
        origmask = createPlantMask(self.im, self.T, self.minSize)
        
        minImg_T = threshold_otsu(self.minImg) / 2
        maskMI = np.array(self.minImg>minImg_T) 
        
        origmask = origmask * maskMI
        
        mask = cleanMask(origmask, self.diskSize);   
        
        return mask
        