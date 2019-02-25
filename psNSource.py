import numpy as np
import copy
import glob
from PSConfig import PSConfig
import os
from psutils import shadowImage, loadIms, loadProperties, saveData, getLSFromRunMode, PGLens, generateWeightMaps
import fcint as fc
import cv2

def psNSource(I, LSorig, imsz):
    LS = copy.deepcopy(LSorig)
    if np.linalg.norm(LS[1]) != 1:
        for idx,i in enumerate(LS):
            LS[idx]=i/np.linalg.norm(i)

    if np.shape(LS)[0]==3:
        LSi=np.linalg.inv(LS);  
    elif np.shape(LS)[0]>3:
        LS_t = np.transpose(LS);
        LSi=np.dot(np.linalg.inv(np.dot(LS_t, LS)),LS_t); 
    AandN=np.dot(LSi, I);
    A=np.sqrt(np.sum(np.abs(AandN)**2,0)); 
    A=A+np.spacing(1) 
    normals=np.divide(AandN, np.kron(np.ones((3,1)),A))
    
    Nx=normals[0]
    Ny=normals[1]
    Nz=normals[2]

    A=A.reshape(imsz);
    Nx=Nx.reshape(imsz); 
    Ny=Ny.reshape(imsz); 
    Nz=Nz.reshape(imsz);
    
    Gx=np.divide(Nx, Nz+np.spacing(1));
    Gy=np.divide(Ny,Nz+np.spacing(1));
    
    return Gx, Gy, Nx, Ny, Nz, A