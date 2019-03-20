'''
    This script displays a GUI to generate rosette masks for the chosen PS-Plant 
    acquisition sessions and desired region of interests (ROI). The GUI allows 
    user to select parameters for better segmentation (threshold, min area, filter
    size).
        
    In the GUI you have to enter:
        - Path to the PS data;
        - Start and end session names from drop down list;
        - ROI file;
        - Plant numbers of interest;
        - Path to the output directory;
        - The user may also preview the output of the chosen parameters (threshold, 
        min area, filter size).


    The Pipeline:
        1. Generate Adaptive Light Source Vectors (GenerateAdaptiveLSVGUI)
        2. Generate SNZShadowImAndAlbedo_adaptiveLS.npz (BatchProcessRawSessionsGUI)
    *   3. Generate Rosette Masks (MaskGenGUI)
        4. Generate Leaf Masks ()
        5. Generate Tracked Leaf Masks (TrackingGUI)
        6. Generate Results (GenerateResultsGUI)
'''
import os
import sys
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QFileDialog, QProgressDialog
import glob
from separateIntoPlots import loadPlant
import psutils as ps
import numpy as np
import cv2
import configparser
import matplotlib.pyplot as plt
from MaskSegmenterBasic import MaskSegmenterBasic

iniFilename='.maskgen.ini'
qtCreatorFile = "MaskGen2.ui" 
    
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
# Defaults 
DEFAULT_ROOT_DIR='D:/Dev/python/Python-PSPlant/Tools'
DEFAULT_ROI_FILE = 'D:/Dev/python/Python-PSPlant/Tools/9plantsroi.txt'
DEFAULT_OUTPUT_DIR = 'D:/Dev/python/Python-PSPlant/Tools/Masks'
DEFAULT_PLANTNOS = '0,2'
DEFAULT_SEGPARAM1 = '70'
DEFAULT_SEGPARAM2 = '300'
DEFAULT_SEGPARAM3 = '11'

DEFAULT_ALBEDO = False
DEFAULT_COMPOSITE = False
DEFAULT_GRAYSCALE = True

class MaskGenGUI(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        # Members
        self.rootDir = DEFAULT_ROOT_DIR
        self.roiFile = DEFAULT_ROI_FILE
        self.outputDir = DEFAULT_OUTPUT_DIR
        self.plants = DEFAULT_PLANTNOS
        self.startDir = None
        self.endDir = None
        self.segParam1 = DEFAULT_SEGPARAM1
        self.segParam2 = DEFAULT_SEGPARAM2
        self.segParam3 = DEFAULT_SEGPARAM3
        self.refreshCombos = True 
        
        self.storeAlbedo = DEFAULT_ALBEDO
        self.storeComposite = DEFAULT_COMPOSITE
        self.storeGrayscale = DEFAULT_GRAYSCALE
        
        self.loadFromIniFile() 
        
        # Buttons
        self.btnRootSel.clicked.connect(self.selectRootDir)
        self.btnOutputSel.clicked.connect(self.selectOutputDir)
        self.btnROISel.clicked.connect(self.selectROIFile)
        self.btnProcess.clicked.connect(self.process)
        self.btnPreview.clicked.connect(self.preview)
        self.rdoVIS.clicked.connect(self.updateLSType)
        self.rdoNIR.clicked.connect(self.updateLSType)
        self.rdoVISNIR.clicked.connect(self.updateLSType)
        self.edtPlants.textChanged.connect(self.updatePlantNos)
         
         
        self.cbCropAlbedo.stateChanged.connect(self.updateAlbedo)
        self.cbCropComposite.stateChanged.connect(self.updateComposite)
        self.cbCropGrayscale.stateChanged.connect(self.updateGrayscale)
         
        self.initVars()
        
        self.cmbStartSess.currentIndexChanged.connect(self.startSessChanged)
        self.cmbEndSess.currentIndexChanged.connect(self.endSessChanged)
        
    def loadFromIniFile(self):
        
        config = configparser.ConfigParser();
        ret = config.read(iniFilename);
        
        if len(ret):
            try:
                self.rootDir = config.get('MaskGen', 'rootDir')
                self.outputDir = config.get('MaskGen', 'outputDir')
                self.roiFile = config.get('MaskGen', 'roiFile')
                self.startDir = config.get('MaskGen', 'startDir')
                self.endDir = config.get('MaskGen', 'endDir')
                self.plants = config.get('MaskGen', 'plants');
                
                self.segParam1 = config.get('SegBasic', 'param1')
                self.segParam2 = config.get('SegBasic', 'param2')
                self.segParam3 = config.get('SegBasic', 'param3')
                
                self.storeAlbedo = config.get('MaskGen', 'storeAlbedo')
                self.storeComposite = config.get('MaskGen', 'storeComposite')
                self.storeGrayscale = config.get('MaskGen', 'storeGrayscale')
                
            except: 
                pass
        
    def saveToIniFile(self):
        config = configparser.ConfigParser();
        config.add_section('MaskGen') 
        config.set('MaskGen', 'rootDir', self.rootDir)
        config.set('MaskGen', 'startDir', self.cmbStartSess.currentText())
        config.set('MaskGen', 'endDir', self.cmbEndSess.currentText())
        config.set('MaskGen', 'roiFile', self.roiFile)
        config.set('MaskGen', 'plants', self.edtPlants.text())
        config.set('MaskGen', 'outputDir', self.outputDir)
        
        config.add_section('SegBasic') 
        config.set('SegBasic', 'param1', self.segParam1)
        config.set('SegBasic', 'param2', self.segParam2)
        config.set('SegBasic', 'param3', self.segParam3)
        
        config.set('MaskGen', 'storeAlbedo', str(int(self.storeAlbedo)))
        config.set('MaskGen', 'storeComposite', str(int(self.storeComposite)))
        config.set('MaskGen', 'storeGrayscale', str(int(self.storeGrayscale)))
        
        with open(iniFilename, 'w') as configfile:
            config.write(configfile)
        
    def initVars(self):
        self.edtRootDir.setText(self.rootDir)
        self.edtROIPath.setText(self.roiFile)
        self.edtOutputDir.setText(self.outputDir)
        
        self.cbCropAlbedo.setChecked(int(self.storeAlbedo))
        self.cbCropComposite.setChecked(int(self.storeComposite))
        self.cbCropGrayscale.setChecked(int(self.storeGrayscale))
        
        if self.refreshCombos:
            LSType = self.getLSType()
            dirs = [os.path.basename(os.path.normpath(x)) for x in glob.glob(os.path.join(self.rootDir, LSType))]
            dirs.sort()
            self.cmbStartSess.clear();self.cmbStartSess.addItems(dirs)
            self.cmbEndSess.clear();self.cmbEndSess.addItems(dirs)
            self.cmbPreviewSess.clear();self.cmbPreviewSess.addItems(dirs)
            
            if self.startDir:
                idx = self.cmbStartSess.findText(self.startDir)
                self.cmbStartSess.setCurrentIndex(idx)
                
            if self.endDir:
                idx = self.cmbEndSess.findText(self.endDir)
                self.cmbEndSess.setCurrentIndex(idx)
            else:
                self.cmbEndSess.setCurrentIndex(self.cmbEndSess.count() - 1 )
                
        self.edtPlants.setText(self.plants)
     
        self.edtSegParam1.setText(self.segParam1)
        self.edtSegParam2.setText(self.segParam2)
        self.edtSegParam3.setText(self.segParam3)
        
    def getLSType(self):
        if self.rdoVIS.isChecked():
            return '*_VIS/'
        elif self.rdoNIR.isChecked():
            return '*_NIR/'
        elif self.rdoVISNIR.isChecked():
            return '*_VISNIR/'
        
    def updatePlantNos(self):
        self.plants = self.edtPlants.text()
        
    def updateLSType(self):
        self.refreshCombos = True 
        self.initVars()
    def selectRootDir(self):
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.rootDir, QtGui.QFileDialog.ShowDirsOnly))
        if pth:
            self.rootDir = pth
            self.refreshCombos = True
            self.initVars()
    
    def selectROIFile(self):
        pth = str(QFileDialog.getOpenFileName(self, "Select ROI file", self.roiFile , "ROI files(*.txt)"))
        if pth:
            self.roiFile = pth
            self.initVars()
    
    def selectOutputDir(self):
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.outputDir, QtGui.QFileDialog.ShowDirsOnly))
        if pth:
            self.outputDir = pth
            self.initVars()

    def startSessChanged(self):
       self.startDir = self.cmbStartSess.currentText()
    
    def endSessChanged(self):
       self.endDir = self.cmbEndSess.currentText()
       
    def updateAlbedo(self):
        self.storeAlbedo = self.cbCropAlbedo.isChecked()
        
    def updateComposite(self):
        self.storeComposite = self.cbCropComposite.isChecked()
        
    def updateGrayscale(self):
        self.storeGrayscale = self.cbCropGrayscale.isChecked()
       
    def preview(self):
        previewFolder = os.path.normpath(os.path.join(self.rootDir, str(self.cmbPreviewSess.currentText())))
        plantNo = int(self.edtPreviewPlant.text())
        
        A, z, snIm, shadowIm=loadPlant(previewFolder, plantNo, self.roiFile) 
        shadowIm, minImg, maxImg = ps.shadowImageFromPth(previewFolder)
        rois=np.loadtxt(self.roiFile, dtype=np.uint16) 
        roi=rois[plantNo]
        minImg=minImg[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        
        self.segParam1 = self.edtSegParam1.text()
        self.segParam2 = self.edtSegParam2.text()
        self.segParam3 = self.edtSegParam3.text()
        
        params = {'im' : A, 'minImg' : minImg, 'T':int(self.segParam1), 'minSize':int(self.segParam2), 'diskSize':int(self.segParam3)}
        segmenter = MaskSegmenterBasic(params)
        mask = segmenter.segmentImage()
        
        plt.close('all')
        plt.figure('Preview')
        plt.subplot(1,2,1); plt.imshow(A);
        plt.subplot(1,2,2); plt.imshow(mask);
        plt.show();
        

    def normaliseRange(self, x):
        if(np.nanmax(x)-np.nanmin(x) != 0):
            normData = (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x)) * 255;
            return normData
        else:
            return np.array(x)
    
    def process(self):
        if not os.path.exists(self.outputDir):
            qm = QtGui.QMessageBox
            ret = qm.question(self,'Create Directory?', "Output mask directory does not exist. Create?", qm.Yes | qm.No)
            
            if ret == qm.Yes:
                os.makedirs(self.outputDir)
        
        if self.cbCropAlbedo.isChecked():
            if not os.path.exists(os.path.join(self.outputDir, "CroppedAlbedo")):
                os.makedirs(os.path.join(self.outputDir, "CroppedAlbedo"))
        
        if self.cbCropGrayscale.isChecked():    
            if not os.path.exists(os.path.join(self.outputDir, "CroppedGrayscale")):
                os.makedirs(os.path.join(self.outputDir, "CroppedGrayscale"))
        
        if self.cbCropComposite.isChecked():    
            if not os.path.exists(os.path.join(self.outputDir, "CroppedComposite")):
                os.makedirs(os.path.join(self.outputDir, "CroppedComposite"))
            
        plantNos = [int(i) for i in self.edtPlants.text().split(',')]
        if not plantNos:
            qm.information(self,'Error', 'Please enter the plants to process separated by commas')
        startFolder = os.path.normpath(os.path.join(self.rootDir, str(self.cmbStartSess.currentText())))
        endFolder = os.path.normpath(os.path.join(self.rootDir, str(self.cmbEndSess.currentText())))
        
        self.segParam1=self.edtSegParam1.text()
        self.segParam2=self.edtSegParam2.text()
        self.segParam3=self.edtSegParam3.text()
        
        d=glob.glob(os.path.join(self.rootDir, self.getLSType()))
        
        d.sort()
        d=[os.path.normpath(p) for p in d]
        
        if startFolder:
            startIdx =d.index(startFolder)
            endIdx =d.index(endFolder)
        
            d = d[startIdx:endIdx+1]
       
        total2Proc = len(d)
       
        progress = QProgressDialog("Creating Masks...", "Abort", 0, total2Proc, self);
        progress.setWindowTitle("Mask Generator")
        progress.setWindowModality(QtCore.Qt.WindowModal);
        try:
            for n,pth in enumerate(d):
                progress.setValue(n)
                if progress.wasCanceled():
                    break;
                for p in plantNos:
                    if progress.wasCanceled():
                        break;
                    print('Processing Plant: {}, {}'.format(p, pth));
                    
                    A, z, snIm, shadowIm=loadPlant(pth, p, self.roiFile)
                    shadowIm, minImg, maxImg = ps.shadowImageFromPth(pth)
                    rois=np.loadtxt(self.roiFile, dtype=np.uint16) 
                    roi=rois[p]
                    minImg=minImg[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                    
                    params = {'im' : A, 'minImg' : minImg, 'T':int(self.segParam1), 'minSize':int(self.segParam2), 'diskSize':int(self.segParam3)}
                    segmenter = MaskSegmenterBasic(params)
                    mask = segmenter.segmentImage()
                    mask.dtype = 'uint8'
                    
                    data = np.load(os.path.join(pth, "SNZShadowImAndAlbedo_adaptiveLS.npz"))
                    snIm = data['snIm']
                    A = data['A']
                    A = self.normaliseRange(A)
                    snIm = self.normaliseRange(snIm)
                    
                    im=glob.glob(os.path.join(pth,'*im[0-9].bmp'))
                    ims = []
                    for i,idx in enumerate(im):
                        ims.append(cv2.imread(idx, 0))
                    
                    arr = np.zeros((np.size(ims[0], 1), np.size(ims[0], 0)),np.float)
                    for im in ims:
                        arr=arr+im/(i + 1)
                        
                    composite = np.zeros((np.size(ims[0], 1), np.size(ims[0], 0), 3))
                    composite[..., 0] = snIm[:,:,0]
                    composite[..., 1] = snIm[:,:,1]
                    composite[..., 2] = A
                    
                    arr = self.normaliseRange(arr)
                    
                    fname = '{}_{}_mask.png'.format(p, os.path.basename(pth))
                    cv2.imwrite(os.path.join(self.outputDir, fname), mask*255)
                    
                    if self.cbCropAlbedo.isChecked():
                        fname = '{}_{}_albedo.png'.format(p, os.path.basename(pth))
                        cv2.imwrite(os.path.join(os.path.join(self.outputDir, "CroppedAlbedo"), fname), self.normaliseRange(A)[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
                    
                    if self.cbCropGrayscale.isChecked():
                        fname = '{}_{}_grayscale.png'.format(p, os.path.basename(pth))
                        cv2.imwrite(os.path.join(os.path.join(self.outputDir, "CroppedGrayscale"), fname), arr[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
                    
                    if self.cbCropComposite.isChecked():
                        fname = '{}_{}_composite.png'.format(p, os.path.basename(pth))
                        cv2.imwrite(os.path.join(os.path.join(self.outputDir, "CroppedComposite"), fname), composite[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
            
            progress.setValue(total2Proc);
            qm = QtGui.QMessageBox
            qm.information(self,'Completed', 'Masks can be found in {}.'.format(self.outputDir))
        except Exception as e:
            progress.cancel()
            qm = QtGui.QMessageBox
            qm.information(self,'Error', '{}'.format(e))
            raise e
            
        
    def closeEvent(self, event):
        self.saveToIniFile()
         
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MaskGenGUI()
    window.show()
    window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    window.activateWindow()
    window.raise_()
    sys.exit(app.exec_())
 