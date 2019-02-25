'''
    This is the GUI for generating leaf segmentations. 
    
    In the GUI user has to enter:
        - Cropped image data directory (provided model is for images size of 512x512)
        - Save directory
        - Path to the pre-trained Mask R-CNN model weights (model trained on grayscale 
        images can be donwloaded from: 
        https://liveuweac-my.sharepoint.com/:f:/g/personal/gytis2_bernotas_live_uwe_ac_uk/EqEuMisdwI5CjVjnfB7rofwBlFWJaAZTrEeuSQUjgSGHjw?e=yYT8TR
    
    The Pipeline:
        1. Generate Adaptive Light Source Vectors (GenerateAdaptiveLSVGUI)
        2. Generate SNZShadowImAndAlbedo_adaptiveLS.npz (BatchProcessRawSessionsGUI)
        3. Generate Rosette Masks (MaskGenGUI)
    *   4. Generate Leaf Masks (LeafSegmentationGUI)
        5. Generate Tracked Leaf Masks (TrackingGUI)
        6. Generate Results (GenerateResultsGUI)
        
'''

import os, datetime
import sys

MASK_DIR = os.path.join(os.getcwd(), 'MaskRCNN')
sys.path.append(MASK_DIR)

from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QFileDialog, QProgressDialog
import glob
import numpy as np
import cv2
import configparser
import time

import tensorflow as tf
import mrcnn.model as modellib
from samples.GigaScience import LeafSegmentationConfig

iniFilename='.LeafSegmentationGUI.ini'
qtCreatorFile = "LeafSegmentationGUI.ui" 
    
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
# Defaults 
DEFAULT_ROOT_DIR='D:/Dev/python/Python-PSPlant/Tools'
DEFAULT_SAVE_DIR = 'D:/Dev/python/Python-PSPlant/Tools'
DEFAULT_WEIGHTS_DIR = 'D:/Dev/python/Python-PSPlant/Tools/NLNTGrayscale.h5'

class LeafSegmentationGUI(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        # Members
        self.rootDir = DEFAULT_ROOT_DIR
        self.saveDir = DEFAULT_SAVE_DIR
        self.weightsDir = DEFAULT_WEIGHTS_DIR
        self.refreshCombos = True
        self.Type = 'grayscale'
        self.cmap = np.array([[192, 57, 43], [243, 156, 18], [26, 188, 156], [41, 128, 185], [142, 68, 173], [44, 62, 80],
                                [127, 140, 141], [17, 75, 95], [2, 128, 144], [228, 253, 225], [69, 105, 144],
                                [244, 91, 105], [91, 192, 235], [253, 231, 76], [155, 197, 61], [229, 89, 52],
                                [250, 121, 33], [124, 82, 47], [86, 15, 94], [38, 63, 77], [1, 52, 55], [319, 29, 82],
                                [220,131,52], [52, 220, 194], [52, 115, 220], [255, 163, 138], [251, 255, 138], [87, 90, 12],
                                [80, 158, 80], [255, 0, 0], [0, 255, 0], [0, 0, 255], [244, 91, 105], [91, 192, 235],
                                [253, 231, 76], [155, 197, 61], [229, 89, 52], [250, 121, 33], [124, 82, 47], [86, 15, 94],
                                [38, 63, 77], [222,97,127], [195,114,88], [69,160,104], [132,54,243], [163,130,58], [94,234,184],
                                [215,187,177], [47,64,102], [36,227,83], [248,57,253], [164,27,80], [78,201,159]], dtype='uint8')
        
        self.loadFromIniFile() 
        
        # Buttons
        self.btnRootSel.clicked.connect(self.selectRootDir)
        self.btnSaveSel.clicked.connect(self.selectSaveDir)
        self.btnWeightsSel.clicked.connect(self.selectWeights)
        self.btnProcess.clicked.connect(self.process)
        
        self.rdoAlbedo.clicked.connect(self.updateType)
        self.rdoComposite.clicked.connect(self.updateType)
        self.rdoGrayscale.clicked.connect(self.updateType)
        
        self.initVars()
  
    def loadFromIniFile(self):
        config = configparser.ConfigParser();
        ret = config.read(iniFilename);
        
        if len(ret):
            try:
                self.rootDir = config.get('LeafSegm', 'rootDir')
                self.saveDir = config.get('LeafSegm', 'saveDir')
                self.weightsDir = config.get('LeafSegm', 'weightsDir')
            except: 
                pass
        
    def saveToIniFile(self):
        config = configparser.ConfigParser();
        config.add_section('LeafSegm') 
        config.set('LeafSegm', 'rootDir', self.rootDir)
        config.set('LeafSegm', 'saveDir', self.saveDir)
        config.set('LeafSegm', 'weightsDir', self.weightsDir)
        
        with open(iniFilename, 'w') as configfile:
            config.write(configfile)
        
    def initVars(self):
        self.edtRootDir.setText(self.rootDir)
        self.edtSaveDir.setText(self.saveDir)
        self.edtWeightsDir.setText(self.weightsDir)
        
        if self.refreshCombos:
            self.Type = self.getType()           
            self.refreshCombos = False
    
    def getType(self):
        if self.rdoGrayscale.isChecked():
            return 'grayscale'
        elif self.rdoAlbedo.isChecked():
            return 'albedo'
        elif self.rdoComposite.isChecked():
            return 'composite'
        
    def updateType(self):
        self.refreshCombos = True 
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
            
    def selectWeights(self):
        pth = str(QFileDialog.getOpenFileName(self, "Select pre-trained model weights (*.h5)", self.weightsDir , "Mask R-CNN model weights (*.h5)"))
        if pth:
            self.weightsDir = pth
            self.initVars()                
            
    def process(self):
        MODEL_DIR = os.path.join(MASK_DIR, "logs")
        config = LeafSegmentationConfig.ArabidopsisLeafSegmentationConfig()

        class InferenceConfig(config.__class__):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        
        config = InferenceConfig()
        config.display()
        DEVICE = "/gpu:0"
        
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        
        model.load_weights(self.weightsDir, by_name=True)
        
        imgs = glob.glob(self.rootDir + "\\*_" + self.Type + "*")
        
        total2Proc = len(imgs)
       
        progress = QProgressDialog("Generating results...", "Abort", 0, total2Proc, self);
        progress.setWindowTitle("Batch Process")
        progress.setWindowModality(QtCore.Qt.WindowModal);
        
        try:
            for i,idx in enumerate(imgs):
                im = cv2.imread(idx)
                results = model.detect([im], verbose=1)
            
                r = results[0]
                
                finalMask = np.zeros([np.size(r['masks'],1), np.size(r['masks'],0)])
                finalMask2 = np.zeros([np.size(r['masks'],1), np.size(r['masks'],0), 3])
                for j in range(0, np.size(r['masks'], 2)):
                    alpha = (j+1) * r['masks'][:,:,j]
                    overlap = np.multiply(finalMask, alpha)
                    finalMask = finalMask + alpha
                    finalMask[overlap > 0] = np.max(alpha)
                    
                for j in range(0,  np.size(r['masks'],2)):
                    finalMask2[finalMask == (j + 1), 0] =  cmap[j][0]
                    finalMask2[finalMask == (j + 1), 1] =  cmap[j][1]
                    finalMask2[finalMask == (j + 1), 2] =  cmap[j][2]
                
                name = idx.split("\\")[-1].split(self.Type)[0]
                savePath = self.saveDir + "\\{}mask.png".format(name)
                cv2.imwrite(savePath, finalMask2);    
                progress.setValue(i);
            progress.setValue(total2Proc);
            qm = QtGui.QMessageBox
            qm.information(self,'Completed', 'Label images were created.')
            
        except Exception as e:
            progress.cancel()
            qm = QtGui.QMessageBox
            qm.information(self,'Error', '{}'.format(e))
            raise e
            
    def closeEvent(self, event):
        self.saveToIniFile()
         
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = LeafSegmentationGUI()
    window.show()
    window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    window.activateWindow()
    window.raise_()
    sys.exit(app.exec_())
 