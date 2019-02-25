'''
    This is the GUI for processing raw files and generating PS outputs that are
    stored in SNZShadowImAndAlbedo_adaptiveLS.npz files. 
    
    In the GUI user has to enter:
        - Data directory and light source type (VIS, NIR or VISNIR);
        - Start and end session names;
        - Whether to use or not the adaptive light source vectors (if yes, 
        provide the path to the adaptive light source vectors).
    
    The Pipeline:
        1. Generate Adaptive Light Source Vectors (GenerateAdaptiveLSVGUI)
    *   2. Generate SNZShadowImAndAlbedo_adaptiveLS.npz (BatchProcessRawSessionsGUI)
        3. Generate Rosette Masks (MaskGenGUI)
        4. Generate Leaf Masks ()
        5. Generate Tracked Leaf Masks (TrackingGUI)
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

iniFilename='.batchprocessrawsessions.ini'
qtCreatorFile = "BatchProcessRawSessionsGUI.ui" 
    
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
# Defaults 
DEFAULT_ROOT_DIR='D:/Dev/python/Python-PSPlant/Tools'
DEFAULT_LSVFILE = 'D:/Dev/python/Python-PSPlant/Tools/AdaptiveLSVsFirstAfterNIRFilter.npz'

class BatchProcessRawSessionsGUI(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        # Members
        self.rootDir = DEFAULT_ROOT_DIR
        self.adaptiveLSVFile = DEFAULT_LSVFILE
        self.startDir = None
        self.endDir = None
        self.refreshCombos = True
        self.useAdaptivePS = True
        
        self.loadFromIniFile() 
        
        # Buttons
        self.btnRootSel.clicked.connect(self.selectRootDir)
        self.btnAdaptiveLSVFileSel.clicked.connect(self.selectAdaptiveLSVFile)
        self.btnProcess.clicked.connect(self.process)
        self.rdoVIS.clicked.connect(self.updateLSType)
        self.rdoNIR.clicked.connect(self.updateLSType)
        self.rdoVISNIR.clicked.connect(self.updateLSType)
        
        self.cbUseAdaptivePS.stateChanged.connect(self.updateAdaptivePS)
        self.initVars()
        
        self.cmbStartSess.currentIndexChanged.connect(self.startSessChanged)
        self.cmbEndSess.currentIndexChanged.connect(self.endSessChanged)
       
  
    def loadFromIniFile(self):
        config = configparser.ConfigParser();
        ret = config.read(iniFilename);
        
        if len(ret):
            try:
                self.rootDir = config.get('BatchProc', 'rootDir')
                self.adaptiveLSVFile = config.get('BatchProc', 'adaptiveLSVFile')
                self.startDir = config.get('BatchProc', 'startDir')
                self.endDir = config.get('BatchProc', 'endDir')
                self.useAdaptivePS = json.loads(config.get('BatchProc', 'useAdaptivePS'))
            except: 
                pass
        
    def saveToIniFile(self):
        config = configparser.ConfigParser();
        config.add_section('BatchProc') 
        config.set('BatchProc', 'rootDir', self.rootDir)
        config.set('BatchProc', 'startDir', self.cmbStartSess.currentText())
        config.set('BatchProc', 'endDir', self.cmbEndSess.currentText())
        config.set('BatchProc', 'useAdaptivePS', str(int(self.useAdaptivePS)))
        config.set('BatchProc', 'adaptiveLSVFile', self.adaptiveLSVFile)
        
        with open(iniFilename, 'w') as configfile:
            config.write(configfile)
        
    def initVars(self):
        self.edtRootDir.setText(self.rootDir)
        self.edtAdaptiveLSVFile.setText(self.adaptiveLSVFile)
        self.cbUseAdaptivePS.setChecked(self.useAdaptivePS)
        
        if self.refreshCombos:
            LSType = self.getLSType()
            dirs = [os.path.basename(os.path.normpath(x)) for x in glob.glob(os.path.join(self.rootDir, LSType))]
            dirs.sort()
            self.cmbStartSess.clear();self.cmbStartSess.addItems(dirs)
            self.cmbEndSess.clear();self.cmbEndSess.addItems(dirs)
            self.refreshCombos = False
            
            if self.startDir:
                idx = self.cmbStartSess.findText(self.startDir)
                self.cmbStartSess.setCurrentIndex(idx)
                
            if self.endDir:
                idx = self.cmbEndSess.findText(self.endDir)
                self.cmbEndSess.setCurrentIndex(idx)
            else:
                self.cmbEndSess.setCurrentIndex(self.cmbEndSess.count() - 1 )
    
    def getLSType(self):
        if self.rdoVIS.isChecked():
            return '*_VIS/'
        elif self.rdoNIR.isChecked():
            return '*_NIR/'
        elif self.rdoVISNIR.isChecked():
            return '*_VISNIR/'
        
    def startSessChanged(self):
       self.startDir = self.cmbStartSess.currentText()
    
    def endSessChanged(self):
       self.endDir = self.cmbEndSess.currentText()
 
    def getOverrideLSType(self):
        if self.rdoOverrideVIS.isChecked():
            return 'VIS'
        elif self.rdoOverrideNIR.isChecked():
            return 'NIR'
        elif self.rdoOverrideVISNIR.isChecked():
            return 'VISNIR'
        elif self.rdoOverrideNone.isChecked():
            return None
            
    def updateLSType(self):
        self.refreshCombos = True 
        self.initVars()
        
    def updateAdaptivePS(self):
        self.useAdaptivePS = self.cbUseAdaptivePS.isChecked()
        self.edtAdaptiveLSVFile.setEnabled(self.useAdaptivePS)
        self.grpBoxLSOverride.setEnabled(self.useAdaptivePS)
        
    def selectRootDir(self):
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.rootDir, QtGui.QFileDialog.ShowDirsOnly))
        if pth:
            self.rootDir = pth
            self.refreshCombos = True
            self.initVars()
    
    def selectAdaptiveLSVFile(self):
        pth = str(QFileDialog.getOpenFileName(self, "Select Adaptive LS file", self.adaptiveLSVFile , "Adaptive LSV files(*.npz)"))
        if pth:
            self.adaptiveLSVFile = pth
            self.initVars()
            
    def timeStampString(self):
        s=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return s
    
    def psNSourceBatch(self, dirs, progressBar=None):
        outFname='SNZShadowImAndAlbedo.npz'
        
        for i,d in enumerate(dirs):
            str = 'Processing: {}'.format(d)
            print(str)
            progressBar.setValue(i)
            progressBar.show()
            
            QtGui.QApplication.processEvents()
            time.sleep(0.1)
            if (progressBar.wasCanceled()):
                break;
        
            psconfig=loadProperties(d, 'PSConfig.properties')
            runMode = d.split('_')[-1]
            LS=getLSFromRunMode(runMode, psconfig.LS_VIS, psconfig.LS_NIR, psconfig.LS_VISNIR)
            
            imgs, imsz=loadIms(d, len(LS))
            imgs=np.array(imgs)
            
            if psconfig.applyLightingCompensation:
                lens = PGLens(psconfig.lens)
                weightMaps=generateWeightMaps(np.multiply(LS,10), imsz, lens, [])
                imgs=np.multiply(imgs, weightMaps)
            
            shadowIm,dummy,dummy=shadowImage(imgs);
            shadowIm=np.reshape(shadowIm, imsz)
            
            Gx, Gy, Nx, Ny, Nz, A=psNSource(imgs, LS, imsz)
            im=np.zeros((Nx.shape[0],Nx.shape[1], 3), 'float32')
            im[:,:,0]=Nx
            im[:,:,1]=Ny
            im[:,:,2]=Nz
            z=fc.fcint(Nx, Ny, Nz)
            resFname=os.path.join(d,outFname)
            saveData(resFname, z, A, im, shadowIm)
            
        return len(d)
    
    def psNSourceAdaptiveBatch(self, dirs, pth2AdaptiveLSi, mode='NIR', overrideLightingMode=None, progressBar=None):
        outFname='SNZShadowImAndAlbedo_adaptiveLS.npz'
        data=np.load(pth2AdaptiveLSi)
        LSAdaptedi=data['LSAdaptedi']
        
        if mode=='*_NIR/':
            N=4
        elif mode=='*_VIS/':
            N=4
        else:
            N=8
        
        for i, d in enumerate(dirs):
            str = 'Processing: {}'.format(d)
            print(str)
            progressBar.setValue(i)
            progressBar.setLabelText('This takes a long time. Please be patient...\r\n'+str)
            progressBar.show()
            
            QtGui.QApplication.processEvents()
            time.sleep(0.1)
            if (progressBar.wasCanceled()):
                break;
            imgs, imsz=loadIms(d, N)
            
            ambientIm=[]
            psconfig = PSConfig.fromPropertiesFile(os.path.join(d, 'PSConfig.properties'))
            if psconfig.subtractAmbient:
                try:    
                    ambientFname=os.path.join(d,'imAmbient.bmp')
                    if (os.path.exists(ambientFname)):
                        ambientIm=cv2.imread(ambientFname, cv2.IMREAD_GRAYSCALE)
                except FileNotFoundError:
                    print('No ambient file found for: ' + impath);
            
            if overrideLightingMode is not None:
                if len(imgs) == 8:
                    if overrideLightingMode=='VIS':
                        imgs=imgs[0:4]
                    else:
                        imgs=imgs[4:8]
                        
                if LSAdaptedi.shape[3] == 8:
                    if overrideLightingMode=='VIS':
                        LSAdaptedi = LSAdaptedi[:,:,:,0:4]
                    else:
                        if LSAdaptedi.shape[3] == 8:
                            LSAdaptedi = LSAdaptedi[:,:,:,4:8]
                        
            if np.size(ambientIm) > 0:
                for im in imgs:
                    np.subtract(im, ambientIm.flatten(), im)
                    
                        
            imgs=np.array(imgs)
            
            psconfig=loadProperties(d, 'PSConfig.properties')
            runMode = d.split('_')[-1]
            if overrideLightingMode:
                runMode = overrideLightingMode
                
            LS=getLSFromRunMode(runMode, psconfig.LS_VIS, psconfig.LS_NIR, psconfig.LS_VISNIR)
            lens = PGLens(psconfig.lens) 
            if psconfig.applyLightingCompensation:
                weightMaps=generateWeightMaps(np.multiply(LS,10), imsz, lens, [])
                imgs=np.multiply(imgs, weightMaps)
            
            shadowIm,dummy, dummy=shadowImage(imgs)
            shadowIm=np.reshape(shadowIm, imsz)
            
            Gx, Gy, Nx, Ny, Nz, A=psNSourceAdaptive(imgs, LSAdaptedi, imsz)
            im=np.zeros((Nx.shape[0],Nx.shape[1], 3), 'float32')
            im[:,:,0]=Nx
            im[:,:,1]=Ny
            im[:,:,2]=Nz
            z=fc.fcint(Nx, Ny, Nz)
            resFname=os.path.join(d,outFname)
            saveData(resFname, z, A, im, shadowIm)
            
        return len(dirs)
        
    def process(self):
        startFolder = os.path.normpath(os.path.join(self.rootDir, str(self.startDir)))
        endFolder = os.path.normpath(os.path.join(self.rootDir, str(self.endDir)))
   
        d=glob.glob(os.path.join(self.rootDir, self.getLSType()))
        
        d.sort()
        d=[os.path.normpath(p) for p in d]
        
        if startFolder:
            startIdx =d.index(startFolder)
            endIdx =d.index(endFolder)
        
            d = d[startIdx:endIdx+1]
       
        total2Proc = len(d) 
       
        progress = QProgressDialog("Generating results...", "Abort", 0, total2Proc, self);
        progress.setWindowTitle("Batch Process")
        progress.setWindowModality(QtCore.Qt.WindowModal);
        
        try:
            if self.useAdaptivePS:
            
                pth2AdaptiveLSi = self.adaptiveLSVFile
                self.psNSourceAdaptiveBatch(d, pth2AdaptiveLSi, self.getLSType(), overrideLightingMode=self.getOverrideLSType(), progressBar=progress)
                progress.setValue(total2Proc);

                qm = QtGui.QMessageBox
                qm.information(self,'Completed', 'SNZShadowImAndAlbedo_adaptiveLS.npz files created.')
            
            else:
                self.psNSourceBatch(d, progressBar=progress)
                progress.setValue(total2Proc);
                
                qm = QtGui.QMessageBox
                qm.information(self,'Completed', 'SNZShadowImAndAlbedo.npz files created.')
        except Exception as e:
            progress.cancel()
            qm = QtGui.QMessageBox
            qm.information(self,'Error', '{}'.format(e))
            raise e
            
    def closeEvent(self, event):
        self.saveToIniFile()
         
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = BatchProcessRawSessionsGUI()
    window.show()
    window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    window.activateWindow()
    window.raise_()
    sys.exit(app.exec_())
 