'''
    This script displays a GUI to generate adaptive light source vectors that 
    will be used to generate more accurate 3D representations using PS.
        
    In the GUI you have to enter:
        - Focal length of the camera lens;
        - Sensor size of the camera;
        - Resolution of the camera;
        - Path to the PSConfig.properties file;
        - Path where the adaptive light source file will be stored.

    The Pipeline:
    *   1. Generate Adaptive Light Source Vectors (GenerateAdaptiveLSVGUI)
        2. Generate SNZShadowImAndAlbedo_adaptiveLS.npz (BatchProcessRawSessionsGUI)
        3. Generate Rosette Masks (MaskGenGUI)
        4. Generate Leaf Masks ()
        5. Generate Tracked Leaf Masks (TrackingGUI)
        6. Generate Results (GenerateResultsGUI)
    
'''

import os
import sys
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QFileDialog, QProgressDialog
import numpy as np
import cv2
import configparser
from psutils import getLSFromRunMode, Lens
from PSConfig import PSConfig
import time
from LensSensorFOVCalc import LensSensorPixelCalc
from math import sqrt

iniFilename='.generateadaptivelsv.ini'
qtCreatorFile = "GenerateAdaptiveLSV.ui" 
    
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
# Defaults 
DEFAULT_FOCALLEN='16'
DEFAULT_PSCONFIG_FILE = 'D:/Dev/python/Python-PSPlant/Tools/PSConfig.properties'
DEFAULT_OUTPUT_DIR = 'D:/Dev/python/Python-PSPlant/Tools'
DEFAULT_SENSORSIZE = '8.8x8.8'
DEFAULT_RESOLUTION = '2048x2048'

class GenerateAdaptiveLSVGUI(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        # Members
        self.focalLength = DEFAULT_FOCALLEN
        self.psConfigFile = DEFAULT_PSCONFIG_FILE
        self.outputDir = DEFAULT_OUTPUT_DIR
        self.sensorSize = DEFAULT_SENSORSIZE
        self.resolution = DEFAULT_RESOLUTION
        self.count = 0
        self.loadFromIniFile() 
        
        # Buttons
        self.btnPSConfigSel.clicked.connect(self.selectPSConfig)
        self.btnOutputSel.clicked.connect(self.selectOutputDir)
        self.btnProcess.clicked.connect(self.process)
        self.edtSensorSize.textChanged.connect(self.updateSensorSize)
        self.edtResolution.textChanged.connect(self.updateResolution)
        self.edtFocalLength.textChanged.connect(self.updateFocalLength)
        self.initVars()
        
        
    def getLSType(self):
        if self.rdoVIS.isChecked():
            return 'VIS'
        elif self.rdoNIR.isChecked():
            return 'NIR'
        elif self.rdoVISNIR.isChecked():
            return 'VISNIR'
  
            
    def loadFromIniFile(self):
        config = configparser.ConfigParser();
        ret = config.read(iniFilename);
        
        if len(ret):
            try:
                self.focalLength = config.get('GenerateAdaptiveLSV', 'focalLength')
                self.outputDir = config.get('GenerateAdaptiveLSV', 'outputDir')
                self.psConfigFile = config.get('GenerateAdaptiveLSV', 'psConfigFile')
                self.sensorSize = config.get('GenerateAdaptiveLSV', 'sensorSize')
                self.resolution = config.get('GenerateAdaptiveLSV', 'resolution')
               
            except:
                pass
        
    def saveToIniFile(self):
        config = configparser.ConfigParser();
        config.add_section('GenerateAdaptiveLSV') 
        config.set('GenerateAdaptiveLSV', 'focalLength', self.focalLength)
        config.set('GenerateAdaptiveLSV', 'psConfigFile', self.psConfigFile)
        config.set('GenerateAdaptiveLSV', 'sensorSize', self.sensorSize)
        config.set('GenerateAdaptiveLSV', 'outputDir', self.outputDir)
        config.set('GenerateAdaptiveLSV', 'resolution', self.resolution)
        
        with open(iniFilename, 'w') as configfile:
            config.write(configfile)
        
    def initVars(self):
        self.edtFocalLength.setText(self.focalLength)
        self.edtPSConfigFile.setText(self.psConfigFile)
        self.edtOutputDir.setText(self.outputDir)
        self.edtSensorSize.setText(self.sensorSize)
        self.edtResolution.setText(self.resolution)
   
    def selectPSConfig(self):
        pth = str(QFileDialog.getOpenFileName(self, "Select PSConfig.properties file", self.psConfigFile , "PSConfig files(*.properties)"))
        if pth:
            self.psConfigFile = pth
            self.initVars()
    
    def selectOutputDir(self):
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.outputDir, QtGui.QFileDialog.ShowDirsOnly))
        if pth:
            self.outputDir = pth
            self.initVars()
    
    def updateSensorSize(self):
        self.sensorSize = self.edtSensorSize.text()
        
    def updateResolution(self):
        self.resolution = self.edtResolution.text()
        
    def updateFocalLength(self):
        self.focalLength = self.edtFocalLength.text()
        
    def genAdaptiveLSVectors(self, progress):
        filename='PSConfig.properties'
        psconfig=PSConfig.fromPropertiesFile(self.psConfigFile)
        runMode = self.getLSType()
            
        LS=getLSFromRunMode(runMode, psconfig.LS_VIS, psconfig.LS_NIR, psconfig.LS_VISNIR) 
  
        imsz = [int(s) for s in self.resolution.split('x')]
        sensorSize = [float(s) for s in self.sensorSize.split('x')]
        lens=Lens( int(self.focalLength), sensorSize)

        t = time.time()
        LSAdaptedi, LSAdapted=self.genAdaptiveLSVectorsListComp(LS, imsz, lens, progress) 
        elapsed = time.time() - t
        print('List comp generation: {}' .format(elapsed))
        fname ='AdaptiveLSV_r{}_ss{}_fl{}.npz'.format(self.resolution, self.sensorSize, self.focalLength)
        np.savez_compressed(os.path.join(self.outputDir, fname), LSAdaptedi=LSAdaptedi, LSAdapted=LSAdapted)
        
        return LSAdaptedi, LSAdapted, fname
              
    def genAdaptiveLSVector(self, LS, adjX, adjY, totalCount, progress):
        if self.count % 100000 == 0:
            print('Count {} of {}.'.format(self.count, totalCount))
            if self.count != 0:
                progress.setValue((self.count * 100) / totalCount);
        
        self.count += 1
        
        currLS=np.zeros(LS.shape)
        for idx,L in enumerate(LS):
            currLS[idx, 0]=L[0]+(adjX) 
            currLS[idx, 1]=L[1]+(adjY) 
            currLS[idx, 2]=L[2]
    
        if sqrt(currLS[0,0]**2+currLS[0,1]**2+currLS[0,2]**2) != 1:
            for idx,ls in enumerate(currLS):
                currLS[idx]=ls/sqrt(ls[0]**2+ls[1]**2+ls[2]**2)
        
        if currLS.shape[0]==3:
            LSi=np.linalg.inv(currLS);  
        elif currLS.shape[0]>3:
            LS_t = np.transpose(currLS);
            LSi=np.dot(np.linalg.inv(np.dot(LS_t, currLS)),LS_t);        
        
        adaptiveLSVi=LSi       
        adaptiveLSV=currLS   

        return adaptiveLSVi, adaptiveLSV
        
    def genAdaptiveLSVectorsListComp(self, origLSV, imsz, lens, progress):
        LS=origLSV       
        z=LS[0][2]*10
        pix_w,pix_h=LensSensorPixelCalc(lens.focalLen, [lens.sensorSize[0],lens.sensorSize[1]], z, [imsz[0], imsz[1]])

        midpointX = np.round(imsz[0] / 2)
        midpointY = np.round(imsz[1] / 2)
        adjX = np.arange(imsz[0]-midpointX, 0-(imsz[0]-midpointX), -1)
        adjY = np.arange(imsz[1]-midpointY,0-(imsz[1]-midpointY), -1)
        LS=np.array(LS)
        
        [i,j]=np.meshgrid(np.arange(0,imsz[0]), np.arange(0, imsz[1]))
        i=i.flatten()
        j=j.flatten()    
        
        pix_w/=10
        pix_h/=10
   
        totalCount = imsz[0] * imsz[1]
        M= [self.genAdaptiveLSVector(LS, adjX[ii]*pix_w, adjY[jj]*pix_h, totalCount, progress) for ii,jj in zip(i,j)]
      
        adaptiveLSVi=np.reshape(np.array([m[0] for m in M]), [imsz[1],imsz[0], 3, LS.shape[0]]) 
        adaptiveLSV=np.reshape(np.array([m[1] for m in M]), [imsz[1],imsz[0], LS.shape[0], 3])
       
        return adaptiveLSVi, adaptiveLSV
        
    def process(self):
        self.count = 0
        if not os.path.exists(self.outputDir):
            qm = QtGui.QMessageBox
            ret = qm.question(self,'Create Directory?', "Output directory does not exist. Create?", qm.Yes | qm.No)
            
            if ret == qm.Yes:
                os.makedirs(self.outputDir)
            
        total2Proc = 100
       
        progress = QProgressDialog("Generating Adaptive Light Source Vectors. This can take approx. 5-15 mins.", "Abort", 0, total2Proc, self);
        progress.setWindowTitle("Generate AdaptiveLSV")
        progress.setWindowModality(QtCore.Qt.WindowModal);
        progress.show()
        QtGui.QApplication.processEvents()
        time.sleep(0.1)
        try:
            LSAdaptedi, LSAdapted, fname = self.genAdaptiveLSVectors(progress)
            progress.setValue(total2Proc);
            progress.cancel()
            qm = QtGui.QMessageBox
            qm.information(self,'Completed', 'Results can be found in {}\\{}.'.format(self.outputDir, fname))
        except Exception as e:
            progress.cancel()
            qm = QtGui.QMessageBox
            qm.information(self,'Error', '{}'.format(e))
            raise e
            
        
    def closeEvent(self, event):
        self.saveToIniFile()
         
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = GenerateAdaptiveLSVGUI()
    window.show()
    window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    window.activateWindow()
    window.raise_()
    sys.exit(app.exec_())
 