""" 

AcquirePSImages

This script includes functions for acquiring, processing, exporting and importing 
PS data. The acquired images are stored in archive directory specified in 
PSConfig.properties file. 

"""

import os, datetime
import numpy as np
from psNSource import psNSource
import psutils as psut
from fcint import fcint
import matplotlib.pyplot as plt
import sys
import shutil
from psutils import Lens, getLSFromRunMode
import glob 
from PSConfig import PSConfig
import PSPGCam as cam
from skimage import morphology
from skimage.filters import threshold_otsu
import warnings
import cv2
import scipy.io as sio
from separateIntoPlots import segmentImage
import psNSourceAdaptive

class HaltException(Exception): pass

Fname3DData='3Ddata.mat'

    
def procCmdLine(argv):
    runMode=''
    if len(argv) == 1:
        raise HaltException("Run mode (VIS, NIR, VISNIR) not provided in command line arguments")
    else:
        runMode=argv[1]
    
    validRunModes = ['VIS', 'NIR', 'VISNIR']
    if not any(x in runMode for x in validRunModes):
        raise HaltException('Invalid runMode specified in cmdline. Must be VIS, NIR or VISNIR')
    
    print('Argument List:', str(argv))
    return runMode
    
def loadConfig(filename):
    config=PSConfig.fromPropertiesFile(filename)
    return config
          
def processPSImages(impath, LS, roi=None, psconfig=None, overrideLightingMode=None, sessionFName=None):
    ambientIm=[]
    if psconfig is not None:
        roi=psconfig.roi
        if psconfig.subtractAmbient:
            try:    
                ambientFname=os.path.join(impath,'imAmbient.bmp')
                if (os.path.exists(ambientFname)):
                    ambientIm=cv2.imread(ambientFname, cv2.IMREAD_GRAYSCALE).astype(np.float32);  
            except FileNotFoundError:
                print('No ambient file found for: ' + impath);
    
    imfiles=glob.glob(os.path.join(impath,'*[0-9].bmp'))
    if overrideLightingMode is not None:
        if len(imfiles) == 8: 
            if overrideLightingMode=='VIS':
                imfiles=imfiles[0:4]
            else:
                imfiles=imfiles[4:8]
                
    imgs=[]
    imgs2D=[]
    for i in imfiles:
        im = cv2.imread(i, cv2.IMREAD_GRAYSCALE).astype(np.float32); 
        if np.size(ambientIm) > 0:
            im=np.subtract(im, ambientIm)
            if (np.any(im < 0)):
                nNegs=np.where(im < 0)
                msg='Ambient subtraction has led to {} negative values - correcting...'.format(np.shape(nNegs)[1])
                warnings.warn(msg)
                im[im < 0]=0 
            
        origshape=im.shape
        if not roi is None:
            im=im[roi[2]:roi[3], roi[0]:roi[1]] 
        imgs2D.append(im)
    
    imgs2D=np.array(imgs2D)
    minImg = imgs2D.min(0)
    
    imgs=np.array([im.flatten() for im in imgs2D])
    
    weightMaps=[]
    if psconfig is not None:
        if psconfig.applyLightingCompensation:
            lens = psut.PGLens(psconfig.lens) if psconfig.environment == 'PC' else psut.rPiLens
            print('Using camera/lens with focal length: %dmm and sensor size of %2.2fx%2.2fmm'%(lens.focalLen, lens.sensorSize[0], lens.sensorSize[1]))
            weightMaps=psut.generateWeightMaps(np.multiply(LS,10), origshape, lens, roi) 
            imgs=np.multiply(imgs, weightMaps)
    
    Gx, Gy, Nx, Ny, Nz, A=psNSource(imgs, LS, np.shape(im));
    
    Nx[np.isnan(Nx)]=np.spacing(1)
    Ny[np.isnan(Ny)]=np.spacing(1)
    Nz[np.isnan(Nz)]=1
    
    Nx[Nx==0]=np.spacing(1)
    Ny[Ny==0]=np.spacing(1)
    Nz[Nz==0]=1
    
    z=fcint(Nx, Ny, Nz)
   
    if psconfig is not None:
        if psconfig.crop2Plant:
            mask = segmentImage(A, minImg)
            z[mask==0]=np.nan
    
    return Nx, Ny, Nz, A, z

def createTimeStampDirName(pth, runMode):
    s=os.path.join(pth, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + runMode)
    return s

def export3DData(pth, Nx, Ny, Nz, A, z):
    sio.savemat(os.path.join(pth, Fname3DData), {'Nx':Nx, 'Ny':Ny, 'Nz':Nz, 'z':z, 'A':A})
    
def load3DData(pth):
    r=sio.loadmat(os.path.join(pth,Fname3DData))
    return r['Nx'], r['Ny'], r['Nz'], r['A'], r['z']
    
def acquirePSImages(runMode):
    psconfig=loadConfig('PSConfig.properties')
    LS=getLSFromRunMode(runMode, psconfig.LS_VIS, psconfig.LS_NIR, psconfig.LS_VISNIR)
    if not os.path.isdir(psconfig.toBeProcessedDir):
        os.makedirs(psconfig.toBeProcessedDir)
    if not os.path.isdir(psconfig.archivePath):
        os.makedirs(psconfig.archivePath)
    
    if psconfig.environment == 'PC':
        exit_code = cam.captureImages(psconfig.COMPort, psconfig.toBeProcessedDir, runMode)
        if (exit_code != 0):
             raise HaltException('Error has occured - stopping processing. Check console for details')
    else:
        raise HaltException('Unrecognised config.environment: ' + psconfig.environment)
        
    Nx, Ny, Nz, A, z=processPSImages(psconfig.toBeProcessedDir, LS, psconfig=psconfig);
    
    archiveDir = createTimeStampDirName(psconfig.archivePath, runMode)
    
    shutil.move(psconfig.toBeProcessedDir, archiveDir);
    shutil.copyfile('PSConfig.properties', os.path.join(archiveDir, 'PSConfig.properties'))
        
    if psconfig.export3DData: 
        export3DData(archiveDir, Nx, Ny, Nz, A, z)
        
    return [Nx, Ny, Nz, A, z, archiveDir]

try:
    runMode=''
    if __name__ == "__main__":
        runMode=procCmdLine(sys.argv)
        acquirePSImages(runMode)    
        
except HaltException as h:
    print(h)