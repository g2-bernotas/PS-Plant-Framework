import numpy as np
import cv2
import fcint as fc
from math import sqrt
from PSConfig import PSConfig
import os
import glob
from psutils import shadowImage, loadIms, loadProperties, saveData, getLSFromRunMode, PGLens, generateWeightMaps

def psNSourceAdaptivePL(I, LSi):
    AandN=np.dot(LSi,  I);
    
    a=sqrt(np.sum(np.abs(AandN)**2,0)); 
    a=a+ np.spacing(1) 
    normals=np.divide(AandN, a)
    
    A=a
    Nx=normals[0]
    Ny=normals[1]
    Nz=normals[2]
    
    return A, Nx, Ny, Nz

def psNSourceAdaptive(I, LSAdaptedi, imsz):
    M= [psNSourceAdaptivePL(im, LSi) for im,LSi in zip(I.T, LSAdaptedi.reshape([imsz[0]*imsz[1],LSAdaptedi.shape[2], LSAdaptedi.shape[3]]))]
    
    A=np.reshape(np.array([m[0] for m in M]), imsz)
    Nx=np.reshape(np.array([m[1] for m in M]), imsz)
    Ny=np.reshape(np.array([m[2] for m in M]), imsz)
    Nz=np.reshape(np.array([m[3] for m in M]), imsz)
  
    Gx=np.divide(Nx, Nz+np.spacing(1));
    Gy=np.divide(Ny, Nz+np.spacing(1));
       
    return Gx, Gy, Nx, Ny, Nz, A