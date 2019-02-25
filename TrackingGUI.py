'''
    This is the GUI for tracking leaf movements. 
    
    In the GUI you have to enter:
        - Path to the untracked masks;
        - Path to the output directory;
        - Parameters:
            - Span (defaut 0)
            - Memory (default 20)
            - Max displacement (default 30)
            - Min frames (default 10)
    
    The Pipeline:
        1. Generate Adaptive Light Source Vectors (GenerateAdaptiveLSVGUI)
        2. Generate SNZShadowImAndAlbedo_adaptiveLS.npz (BatchProcessRawSessionsGUI)
        3. Generate Rosette Masks (MaskGenGUI)
        4. Generate Leaf Masks ()
    *   5. Generate Tracked Leaf Masks (TrackingGUI)
        6. Generate Results (GenerateResultsGUI)
'''

import os, datetime
import sys
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QFileDialog, QProgressDialog
import glob
import numpy as np
import cv2
import configparser
import json
from psutils import shadowImage, loadIms, loadProperties, saveData, getLSFromRunMode, PGLens, generateWeightMaps
from PSConfig import PSConfig
from psNSourceAdaptive import psNSourceAdaptive
from psNSource import psNSource
import fcint as fc
import time
import pandas as pd
import trackpy as tp
import trackpy.predict
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import label, regionprops
from PIL import Image

cmap = np.array([[192, 57, 43], [243, 156, 18], [26, 188, 156], [41, 128, 185], [142, 68, 173], [44, 62, 80],
                                [127, 140, 141], [17, 75, 95], [2, 128, 144], [228, 253, 225], [69, 105, 144],
                                [244, 91, 105], [91, 192, 235], [253, 231, 76], [155, 197, 61], [229, 89, 52],
                                [250, 121, 33], [124, 82, 47], [86, 15, 94], [38, 63, 77], [1, 52, 55], [319, 29, 82],
                                [220,131,52], [52, 220, 194], [52, 115, 220], [255, 163, 138], [251, 255, 138], [87, 90, 12],
                                [80, 158, 80], [255, 0, 0], [0, 255, 0], [0, 0, 255], [222,97,127], [195,114,88], [69,160,104], [132,54,243], [163,130,58], 
                                [94,234,184],[215,187,177], [47,64,102], [36,227,83], [248,57,253], [164,27,80], [78,201,159], [187, 17, 77], 
                                [113, 134, 97], [64, 143, 31], [0, 30, 91], [172, 97, 108], [149, 236, 84], [161, 64, 190], [120, 160, 235],
                                [94, 2, 230], [112, 96, 25], [59, 0, 243], [133, 57, 8], [220, 206, 119], [55, 20, 170], [100, 39, 137],
                                [19, 99, 10], [161, 126, 162], [220, 0, 129], [67, 87, 124], [232, 61, 51]], dtype='uint8')
                                
iniFilename='.Tracking.ini'
qtCreatorFile = "Tracking.ui"
    
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
# Defaults 
DEFAULT_ROOT_DIR='D:/Dev/python/Python-PSPlant/Tools'
DEFAULT_SAVE_DIR = 'D:/Dev/python/Python-PSPlant/Tools'
DEFAULT_SPAN = 0
DEFAULT_MEMORY = 20
DEFAULT_MAX_DISPLACEMENT = 30
DEFAULT_MIN_FRAMES = 10
DEFAULT_PKLFILE = 'D:/Dev/python/Python-PSPlant/Tools/features.pkl'

class Tracking(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        # Members
        self.rootDir = DEFAULT_ROOT_DIR
        self.saveDir = DEFAULT_SAVE_DIR 
        self.refreshCombos = True
        self.span = DEFAULT_SPAN
        self.memory = DEFAULT_MEMORY
        self.maxDisp = DEFAULT_MAX_DISPLACEMENT
        self.minFrames = DEFAULT_MIN_FRAMES
        self.UsePKL = True
        self.PKLFile = DEFAULT_PKLFILE
        
        self.loadFromIniFile() 
        
        # Buttons
        self.btnRootSel.clicked.connect(self.selectRootDir)
        self.btnSaveSel.clicked.connect(self.selectSaveDir)
        self.btnProcess.clicked.connect(self.process)
        self.btnPKLFileSel.clicked.connect(self.selectPKLFile) ## 
        
        self.cbUsePKL.stateChanged.connect(self.updatePKL) ## 
        
        self.edtMaxDisplacement.textChanged.connect(self.updateMaxDisplacement)
        self.edtMemory.textChanged.connect(self.updateMemory)
        self.edtMinFrames.textChanged.connect(self.updateMinFrames)
        self.edtSpan.textChanged.connect(self.updateSpan)
        
        self.initVars()
  
    def loadFromIniFile(self):
        config = configparser.ConfigParser();
        ret = config.read(iniFilename);
        
        if len(ret):
            try:
                self.rootDir = config.get('Tracking', 'rootDir')
                self.saveDir = config.get('Tracking', 'saveDir')
                self.span = config.get('Tracking', 'span')
                self.memory = config.get('Tracking', 'memory')
                self.maxDisp = config.get('Tracking', 'maxDisp')
                self.minFrames = config.get('Tracking', 'minFrames')
                self.UsePKL = config.get('Tracking', 'UsePKL')
                self.PKLFile = config.get('Tracking', 'PKLFile')
            except: 
                pass
        
    def saveToIniFile(self):
        config = configparser.ConfigParser();
        config.add_section('Tracking') 
        config.set('Tracking', 'rootDir', self.rootDir)
        config.set('Tracking', 'saveDir', self.saveDir)
        config.set('Tracking', 'span', self.span)
        config.set('Tracking', 'memory', self.memory)
        config.set('Tracking', 'maxDisp', self.maxDisp)
        config.set('Tracking', 'minFrames', self.minFrames)
        config.set('Tracking', 'UsePKL', int(self.UsePKL))
        config.set('Tracking', 'PKLFile', self.PKLFile)
        
        with open(iniFilename, 'w') as configfile:
            config.write(configfile)
        
    def initVars(self):
        self.edtRootDir.setText(self.rootDir)
        self.edtSaveDir.setText(self.saveDir)
        self.edtPKLFile.setText(self.PKLFile)
        self.cbUsePKL.setChecked(int(self.UsePKL))
        
        self.edtSpan.setText(str(self.span))
        self.edtMemory.setText(str(self.memory))
        self.edtMaxDisplacement.setText(str(self.maxDisp))
        self.edtMinFrames.setText(str(self.minFrames))
        
    def updatePKL(self):
        self.UsePKL = self.cbUsePKL.isChecked()
        self.edtPKLFile.setEnabled(self.UsePKL)
        self.btnPKLFileSel.setEnabled(self.UsePKL)
        
    def selectPKLFile(self):
        pth = str(QFileDialog.getOpenFileName(self, "Select PKL file", self.PKLFile , "PKL files (features.pkl)"))
        if pth:
            self.PKLFile = pth
            self.initVars()
        
    def selectRootDir(self):
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.rootDir, QtGui.QFileDialog.ShowDirsOnly))
        if pth:
            self.rootDir = pth
            self.refreshCombos = True
            self.initVars()
    
    def selectSaveDir(self):
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.saveDir, QtGui.QFileDialog.ShowDirsOnly))
        if pth:
            self.saveDir = pth
            self.refreshCombos = True
            self.initVars()
    
    def updateMaxDisplacement(self):
        self.maxDisp = self.edtMaxDisplacement.text()
        
    def updateMemory(self):
        self.memory = self.edtMemory.text()
        
    def updateMinFrames(self):
        self.minFrames = self.edtMinFrames.text()
        
    def updateSpan(self):
        self.span = self.edtSpan.text()
    
    def particleTracker(self, dirs, progressBar=None):
        features=pd.DataFrame()
        
        if self.UsePKL:
            features = pd.read_pickle(self.PKLFile)
        else:
            for i,idx in enumerate(dirs):
                progressBar.setValue((i*100/len(dirs))/2)
                progressBar.setLabelText('Processing particles...')
                progressBar.show()
                
                QtGui.QApplication.processEvents()
                if (progressBar.wasCanceled()):
                    break;            
                
                im=cv2.imread(idx, 0)
                masks = []
                
                for j in range(1,255):
                    tmp = np.zeros([np.size(im, 1), np.size(im, 0)])
                    tmp[im == j] = 1 
                    tmp = binary_opening(tmp) 
                    if(np.sum(tmp) > 50):
                        masks.append(tmp)
            
                mask = np.zeros([np.size(im, 1), np.size(im, 0)])
                for j in range(0, np.size(masks, 0)):
                    alpha = (j+1) * masks[j]
                    overlap = np.multiply(mask, alpha)
                    mask = mask + alpha
                    mask[overlap > 0] = np.max(alpha)
                
                l = label(mask)
                rp = regionprops(l)
            
                for region in rp:
                    features = features.append([{'area': region.area,
                                                'y': region.centroid[0],
                                                'x': region.centroid[1],
                                                'frame': i,
                                                },])
            pd.to_pickle(features, self.saveDir + "\\features.pkl")
        
        pred = tp.predict.NearestVelocityPredict(span=int(self.span))
        t= pred.link_df(features, int(self.maxDisp), memory=int(self.memory))
        t1 = tp.filter_stubs(t, int(self.minFrames))
        
        oldFrame = t1.iloc[0][1]
        frame = t1.iloc[0][1]
        j = 0
        
        for i, idx in enumerate(dirs):
            if self.UsePKL:
                progressBar.setValue(i*100/len(dirs))
            else:
                progressBar.setValue(50 + (i*100/len(dirs))/2)
                
            progressBar.setLabelText('Generating new labels...')
            progressBar.show()
            
            QtGui.QApplication.processEvents()
            if (progressBar.wasCanceled()):
                break;
                
            leaves=cv2.imread(idx, 0)
            im = np.zeros([np.size(leaves, 1), np.size(leaves, 0), 3])
            
            while(frame == oldFrame):
                colour = leaves[int(t1.iloc[j][3]),int(t1.iloc[j][2])]
                x = int(t1.iloc[j][3])
                y = int(t1.iloc[j][2])
                min = 1
                while colour == 0:
                    for p in range(-min, min+1, 1):
                        for c in range(-min, min+1, 1):
                            colour = leaves[x + p, y + c]
                            if colour != 0:
                                break
                        if colour != 0:
                            break
                    min = min + 1
                            
                im[leaves == colour] = cmap[int(t1.iloc[j][4])]
                
                j = j + 1
                if j < len(t1):
                    frame = t1.iloc[j][1]
                else:
                    break;
            
            oldFrame = frame
            
            cv2.imwrite(os.path.join(self.saveDir, idx.split("\\")[-1]), im)

    def process(self):
        dirs = sorted(glob.glob(self.rootDir+'\\*_mask*')) 
        total2Proc = 100 
        
        progress = QProgressDialog("Tracking leaves...", "Abort", 0, total2Proc, self);
        progress.setWindowTitle("Batch Process")
        progress.setWindowModality(QtCore.Qt.WindowModal);
        
        try:
            self.particleTracker(dirs, progressBar=progress)
            progress.setValue(total2Proc);
            
            qm = QtGui.QMessageBox
            qm.information(self,'Completed', 'Leaf labels were changed.')
            
        except Exception as e:
            progress.cancel()
            qm = QtGui.QMessageBox
            qm.information(self,'Error', '{}'.format(e))
            raise e
            
    def closeEvent(self, event):
        self.saveToIniFile()
         
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = Tracking()
    window.show()
    window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    window.activateWindow()
    window.raise_()
    sys.exit(app.exec_())
 