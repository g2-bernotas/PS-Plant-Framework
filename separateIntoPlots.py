import numpy as np
import cv2
from retrying import retry
from fcint import fcint
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects, disk, dilation
from skimage.measure import label, regionprops
from psutils import loadIms
import os
import math

def shadowImageFromPth(previewFolder):
    imgs, imsz = loadIms(previewFolder, 4)
    minImg=np.asarray(imgs).min(0)
    maxImg=np.asarray(imgs).max(0)
    shadowIm=(minImg/(maxImg+np.spacing(1)))
    minImg = np.reshape(minImg, (imsz[0], imsz[1]))
    maxImg = np.reshape(minImg, (imsz[0], imsz[1]))
    shadowImg = np.reshape(minImg, (imsz[0], imsz[1]))
    
    return shadowIm, minImg, maxImg

@retry(stop_max_attempt_number=5, wait_fixed=3000) 
def segmentImages(pth, rois, adaptedLSVi):
    imgs=[]
    
    for i in np.arange(0,adaptedLSVi.shape[3]):
        im=cv2.imread(pth + '/im' + str(i) +'.bmp', cv2.IMREAD_GRAYSCALE)
        imsz=im.shape
        imgs.append(im)
    
    imgs=np.array(imgs)
    imagesAndLSVs=[]
    for r in rois:
        imsROI=imgs[:,r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        LSVsROI=adaptedLSVi[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        imagesAndLSVs.append([imsROI, LSVsROI])
        
    return imagesAndLSVs

def createPlantMask(A, T=70, minSize=300):
    binary = A > T
   
    binary=remove_small_objects(binary, minSize)
    binary=remove_small_holes(binary)
   
    return binary

def crop2Plant(A, x):
    thresh = threshold_otsu(A)
    binary = A > 70
   
    binary=remove_small_objects(binary, 300)
    binary=remove_small_holes(binary)
    imageLabels = label(binary, background=0)
    x[imageLabels==0]=np.nan
    return x

def cleanMask(mask, diskSize=11):
    selem = disk(diskSize)
    maskDil=dilation((mask>0).astype(np.uint8), selem) > 0
    label_img = label(maskDil)
    props = regionprops(label_img)
    d=[];
    if len(props) > 1:
        imCentre = [mask.shape[0] / 2, mask.shape[1] /2]
        for i, p in enumerate(props):
            d.append(math.hypot(p.centroid[0] - imCentre[0], p.centroid[1] - imCentre[1]))
            
        idx = np.argmin(d)
        props = props[idx]
        mask = (label_img == props.label) * mask
        mask = mask > 0
    
    return mask

def segmentImage(im, minImg, T=30, minSize=300, diskSize=11):
    origmask = createPlantMask(im, T, minSize)
    minImg_T = threshold_otsu(minImg) / 2
    maskMI = np.array(minImg>minImg_T)    
    origmask = origmask * maskMI
    mask = cleanMask(origmask, diskSize);   
    
    mask = createPlantMask(mask, False, minSize)
    
    return mask

DEFAULT_ROI_FILENAME = '9plantsroi.txt'
@retry(stop_max_attempt_number=5, wait_fixed=3000) 
def loadPlant(pth, plantNo, roiPth):
    if '.txt' not in roiPth:
        roifile=os.path.join(roiPth, DEFAULT_ROI_FILENAME)
    else:
        roifile = roiPth
    rois=np.loadtxt(roifile, dtype=np.uint16) 
    roi=rois[plantNo]
    data=np.load(os.path.join(pth, 'SNZShadowImAndAlbedo_adaptiveLS.npz'))
        
    shadowIm = data['shadowIm']
    snIm = data['snIm']
    A = data['A']
    
    data.close()
    
    snIm=snIm[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2],:]
    A=A[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    shadowIm=shadowIm[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    z=fcint(snIm[:,:,0], snIm[:,:,1], snIm[:,:,2])
    z=crop2Plant(A, z)
   
    shadowIm, minImg, maxImg = shadowImageFromPth(pth)
    minImg = minImg[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
   
    mask = segmentImage(A, minImg)
    z[mask==0]=np.nan
    
    return A, z, snIm, shadowIm