""" 

PSGUI
PS-Plant data demonstration. Interactive PSGUI allows to:
    * Work with acquired acquisitions ('From archive') or capture a new acquisition (1);
    * Choose illumination acquisition type (visible (VIS), near-infrared (NIR) 
    or both (VISNIR)) (2); 
    * Select image processing options (3):
        * 'Light compensation' - enables inverse square law compensation;
        * 'Crop to plant only' - crops the plant from the image;
        * 'Subtract ambient' - subtracts ambient image from captured images;
        * 'Region of interest' - working region
    
The GUI displays integrated 3D surface using Frankot and Chellappa (1988) (4), 
surface normal directions in x, y and z directions (5) and albedo image (6).

The acquired images are first stored in the 'tobeprocessedpath' directory that are 
then copied to the 'archivepath' directory. Both paths are specified in 'PSConfig.properties' 
file.

Modified: 22/11/2018

"""
from PyQt4.uic import loadUiType
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import glob
import visvis as vv
from PyQt4 import QtGui, QtCore
backend = 'pyqt4'
import AcquirePSImages as acqps
from PyQt4.QtGui import QFileDialog
import visvis as vv
from PSConfig import PSConfig
from matplotlib.widgets import RectangleSelector
import shutil
import os

Ui_MainWindow, QMainWindow = loadUiType('window3.ui')

plt.style.use('classic') 
app = vv.use(backend)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        FigureVV = app.GetFigureClass()
        self.fig = FigureVV(self)
        self.fig2 = Figure()
        self.mplvl2.addWidget(self.fig._widget)
        self.canvas = FigureCanvas(self.fig2)
        self.mplvl3.addWidget(self.canvas)

        self.pbProgress.setValue(0)
        self.pbProgress.setVisible(False)
        self.lblProgress.setText('Ready')
        
        self.btnFromDir.pressed.connect(self.processFromArchive)
        self.btnCamera.pressed.connect(self.acquirePSImages)
        self.btnRecalc.pressed.connect(self.reProcess)
        self.btnExport.pressed.connect(self.export)
       
        self.currentArchive='' 
        app_icon = QtGui.QIcon('brl_logo.gif')
        self.setWindowIcon(app_icon)
        
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.activateWindow()
        self.raise_()
    
    def updateProgressBar(self, text, percent):
        if percent > 0:
            self.pbProgress.setVisible(True)
        else:
            self.pbProgress.setVisible(False)
        
        self.pbProgress.setValue(percent)
        self.lblProgress.setText(text)
        app.ProcessEvents() 
            
    def getSelectedMode(self):
        runMode=''
        if self.rbVIS.isChecked():
            runMode='VIS'
        elif self.rbNIR.isChecked():
            runMode='NIR'
        elif self.rbVISNIR.isChecked():
            runMode='VISNIR'
        else:
            print("Unknown Selection radioButton selected");
            
        return runMode
        
    def export(self):
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.currentArchive, QtGui.QFileDialog.ShowDirsOnly))
        if pth:
            runMode=self.getRunModeFromDirName(self.currentArchive)
            exportDirName=acqps.createTimeStampDirName(pth, runMode)
            os.makedirs(exportDirName)
            acqps.export3DData(exportDirName, self.Nx, self.Ny, self.Nz, self.A, self.z)
            filename='PSConfig.properties'
            psconfig=PSConfig.fromPropertiesFile(os.path.join(self.currentArchive, filename))
            guiConfig = self.getGIUConfigProps() 
            if not guiConfig is None:
                psconfig.overwriteWithGuiProps(guiConfig)
                
            psconfig.saveProps(exportDirName);
            bmps=glob.glob(os.path.join(self.currentArchive, 'im*.bmp'))
            for im in bmps:
                shutil.copy(im, os.path.join(exportDirName,'')) 
                
            self.currentArchive=exportDirName
            
            QtGui.QMessageBox.information(self, "Export", "Data has been exported to: %s" % (exportDirName))

    def acquirePSImages(self):
        runMode = self.getSelectedMode()
        self.Nx, self.Ny, self.Nz, self.A,self.z, archiveDir=acqps.acquirePSImages(runMode)
        self.currentArchive=archiveDir
        self._Plot2();
        filename='PSConfig.properties'
        psconfig=PSConfig.fromPropertiesFile(os.path.join(archiveDir, filename)) 
        self.updateConfigPropertiesGroupBox(psconfig)
    
    def updateConfigPropertiesGroupBox(self, psconfig):
        self.grpConfigProperties.setEnabled(True)
        self.chkBoxLightingComp.setChecked(psconfig.applyLightingCompensation)
        self.chkBoxCrop2Plant.setChecked(psconfig.crop2Plant)
        self.chkBoxSubtractAmbient.setChecked(psconfig.subtractAmbient)
        
        self.edtX1.setText(str(psconfig.roi[0]))
        self.edtX2.setText(str(psconfig.roi[1]))
        self.edtY1.setText(str(psconfig.roi[2]))
        self.edtY2.setText(str(psconfig.roi[3]))
         
    def processFromArchive(self):
        filename='PSConfig.properties' 
        psconfig=PSConfig.fromPropertiesFile(filename) 
        startDir =  psconfig.archivePath
        if not self.currentArchive=='':
            startDir=self.currentArchive
            
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory", startDir, QtGui.QFileDialog.ShowDirsOnly))
        if pth:
            self.currentArchive=pth
            if os.path.isfile(os.path.join(pth, acqps.Fname3DData)) :
                self.updateProgressBar('Loading from previously exported 3D data file', 33)
                self.Nx, self.Ny, self.Nz, self.A, self.z = acqps.load3DData(pth)
                self.updateProgressBar('Preparing display ', 66)
                self._Plot2()
                self.updateProgressBar('Done ', 0)
                psconfig3D=PSConfig.fromPropertiesFile(os.path.join(pth, filename)) 
                self.updateConfigPropertiesGroupBox(psconfig3D)
            else:
                if not os.path.isfile(os.path.join(pth, 'im0.bmp')): 
                    QtGui.QMessageBox.information(self, "Invalid Archive Path", "The selected directory does not appear to be a valid archive.")
                else:
                    self.processImages(pth);
    
    def getGIUConfigProps(self):
        applyLightingCompensation=self.chkBoxLightingComp.isChecked() 
        crop2Plant= self.chkBoxCrop2Plant.isChecked()
        subtractAmbient= self.chkBoxSubtractAmbient.isChecked() 
        x1=int(self.edtX1.text())
        x2=int(self.edtX2.text())
        y1=int(self.edtY1.text())
        y2=int(self.edtY2.text())
        guiConfig=PSConfig.fromGUI(applyLightingCompensation, crop2Plant, subtractAmbient, [x1, x2, y1, y2]) 
        return guiConfig
        
    def reProcess(self):
        guiConfig = self.getGIUConfigProps()
        self.processImages(self.currentArchive, guiConfig)
    
    def getRunModeFromDirName(self, pth):
        runMode = pth.split('_')[-1] 
        return runMode      
        
    def processImages(self, pth, guiConfig=None): 
        runMode=self.getRunModeFromDirName(pth)
        
        filename='PSConfig.properties'
        psconfig=PSConfig.fromPropertiesFile(os.path.join(pth, filename))
        
        if not guiConfig is None:
            psconfig.overwriteWithGuiProps(guiConfig)
        
        LS=acqps.getLSFromRunMode(runMode, psconfig.LS_VIS, psconfig.LS_NIR, psconfig.LS_VISNIR)
        self.updateProgressBar('Processing images', 33)
        
        self.Nx, self.Ny, self.Nz, self.A,self.z=acqps.processPSImages(pth, LS, psconfig=psconfig)
        
        self.updateProgressBar('Preparing display', 66)
        self._Plot2()
        
        self.updateProgressBar('Done', 0)
        self.updateConfigPropertiesGroupBox(psconfig)
        
    def _Plot2(self):
        vv.figure(self.fig.nr)
        vv.clf()
        ax1f2 = vv.cla()
        ax1f2.daspect = 1,-1,1
        vv.surf(self.z)
        vv.axis('off')
        
        ax1f1 = self.fig2.add_subplot(221)
        ax1f1.imshow(self.Nx)   
        ax1f1.axis('off')
        
        ax2f1 = self.fig2.add_subplot(222)
        ax2f1.imshow(self.Ny)
        ax2f1.axis('off')
        
        ax3f1 = self.fig2.add_subplot(223)
        ax3f1.imshow(self.Nz)
        ax3f1.axis('off')
        
        ax4f1 = self.fig2.add_subplot(224)
        ax4f1.imshow(self.A, cmap='gray')
        ax4f1.axis('off')
        
        self.canvas.draw()
        if not hasattr(self, 'toolbar'):
            self.toolbar = NavigationToolbar(self.canvas, 
                self.mplwindow3, coordinates=True)
            self.mplvl3.addWidget(self.toolbar)
        
        self.setWindowTitle('PS Acquisition Tool - %s' % self.currentArchive)
        
def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

if True:
    app.Create()
    m = MainWindow()
    app.Run()

else:
    qtApp = QtGui.QApplication([''])    
    m = MainWindow()
    qtApp.exec_()
