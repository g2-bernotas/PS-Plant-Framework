'''
    This is the GUI for extracting rosette and leaf details from either rosette or leaf masks. 
    
    In the GUI you have to enter:
        - Path to the raw data;
        - Start and end folders;
        - Region of interest (ROI) file;
        - Path to the mask folder;
        - Plant numbers of interest;
        - Path to outputs;
        - Whether to override PSConfig.properties (camera focal length, sensor 
        size and height)
    
    The Pipeline:
        1. Generate Adaptive Light Source Vectors (GenerateAdaptiveLSVGUI)
        2. Generate SNZShadowImAndAlbedo_adaptiveLS.npz (BatchProcessRawSessionsGUI)
        3. Generate Rosette Masks (MaskGenGUI)
        4. Generate Leaf Masks ()
        5. Generate Tracked Leaf Masks (TrackingGUI)
    *   6. Generate Results (GenerateResultsGUI)
'''
import numpy.lib.scimath as npmath
import os, datetime
import sys
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QFileDialog, QProgressDialog
import glob
from separateIntoPlots import loadPlant

import copy
import psutils as ps
import numpy as np
import cv2
import configparser
import matplotlib.pyplot as plt
import csv
from skimage.morphology import disk, binary_opening, label, remove_small_objects, dilation, erosion
from skimage.measure import label, regionprops, find_contours
import math 
from scipy.spatial.distance import cdist, pdist, squareform
from PIL import Image
from LensSensorFOVCalc import LensSensorPixelCalc
from LeafRegionProps import LeafRegionProps
from PSConfig import PSConfig 
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage.filters import gaussian_filter

iniFilename='.generateresults.ini'
qtCreatorFile = "GenerateResultsGUI.ui" 
    
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
# Defaults 
DEFAULT_ROOT_DIR='D:/Dev/python/Python-PSPlant/Tools'
DEFAULT_ROI_FILE = 'D:/Dev/python/Python-PSPlant/Tools/9plantsroi.txt'
DEFAULT_OUTPUT_DIR = 'D:/Dev/python/Python-PSPlant/Tools/Results'
DEFAULT_PLANTNOS = '0,2'


class GenerateResultsGUI(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        # Members
        self.rootDir = DEFAULT_ROOT_DIR
        self.roiFile = DEFAULT_ROI_FILE
        self.maskDir = None
        self.outputDir = DEFAULT_OUTPUT_DIR
        self.plants = DEFAULT_PLANTNOS
        self.startDir = None
        self.endDir = None
        self.focalLength = None
        self.camHeight = None
        self.sensorSize= None
                
        self.refreshCombos = True 
        self.loadFromIniFile() 
        
        # Buttons
        self.btnRootSel.clicked.connect(self.selectRootDir)
        self.btnOutputSel.clicked.connect(self.selectOutputDir)
        self.btnROISel.clicked.connect(self.selectROIFile)
        self.btnMaskSel.clicked.connect(self.selectMaskDir)
        self.btnProcess.clicked.connect(self.process)
        self.rdoVIS.clicked.connect(self.updateLSType)
        self.rdoNIR.clicked.connect(self.updateLSType)
        self.rdoVISNIR.clicked.connect(self.updateLSType)
        self.edtPlants.textChanged.connect(self.updatePlantNos)
        self.cbOverride.stateChanged.connect(self.updateOverride)
        self.edtSensorSize.textChanged.connect(self.updateSensorSize)
        self.edtCamHeight.textChanged.connect(self.updateCamHeight)
        self.edtFocalLength.textChanged.connect(self.updateFocalLength)
        self.initVars()
        
        self.cmbStartSess.currentIndexChanged.connect(self.startSessChanged)
        self.cmbEndSess.currentIndexChanged.connect(self.endSessChanged)
       
        
    def loadFromIniFile(self):
        
        config = configparser.ConfigParser();
        ret = config.read(iniFilename);
        
        if len(ret):
            try:
                self.rootDir = config.get('GenResults', 'rootDir')
                self.outputDir = config.get('GenResults', 'outputDir')
                self.maskDir = config.get('GenResults', 'maskDir')
                self.roiFile = config.get('GenResults', 'roiFile')
                self.startDir = config.get('GenResults', 'startDir')
                self.endDir = config.get('GenResults', 'endDir')
                self.plants = config.get('GenResults', 'plants');
                self.focalLength = config.get('GenResults', 'focalLen')
                self.sensorSize = config.get('GenResults', 'sensorSize')
                self.camHeight = config.get('GenResults', 'camHeight')
                
            except: 
                pass
        
    def saveToIniFile(self):
        config = configparser.ConfigParser();
        config.add_section('GenResults') 
        config.set('GenResults', 'rootDir', self.rootDir)
        config.set('GenResults', 'startDir', self.cmbStartSess.currentText())
        config.set('GenResults', 'endDir', self.cmbEndSess.currentText())
        config.set('GenResults', 'roiFile', self.roiFile)
        config.set('GenResults', 'maskDir', str(self.maskDir))
        config.set('GenResults', 'plants', self.edtPlants.text())
        config.set('GenResults', 'outputDir', self.outputDir)

        config.set('GenResults', 'focalLen', str(self.focalLength))
        config.set('GenResults', 'sensorSize', str(self.sensorSize))
        config.set('GenResults', 'camHeight', str(self.camHeight))
    
        with open(iniFilename, 'w') as configfile:
            config.write(configfile)
        
    def initVars(self):
        self.edtRootDir.setText(self.rootDir)
        self.edtROIPath.setText(self.roiFile)
        self.edtMaskDir.setText(self.maskDir)
        self.edtOutputDir.setText(self.outputDir)
        self.edtFocalLength.setText(self.focalLength)
        self.edtSensorSize.setText(self.sensorSize)
        self.edtCamHeight.setText(self.camHeight)
        
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
        self.edtPlants.setText(self.plants)

        
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
         
    def updatePlantNos(self):
        self.plants = self.edtPlants.text()
        
    def updateLSType(self):
        self.refreshCombos = True 
        self.initVars()
      
    def updateSensorSize(self):
        self.sensorSize = self.edtSensorSize.text()
        
    def updateCamHeight(self):
        self.camHeight = self.edtCamHeight.text()
        
    def updateFocalLength(self):
        self.focalLength = self.edtFocalLength.text()
        
    def updateOverride(self):
        self.overridePSConfig = self.cbOverride.isChecked()
        self.edtFocalLength.setEnabled(self.overridePSConfig)
        self.edtSensorSize.setEnabled(self.overridePSConfig)
        self.edtCamHeight.setEnabled(self.overridePSConfig)

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
    
    def selectMaskDir(self):
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.outputDir, QtGui.QFileDialog.ShowDirsOnly))
        if pth:
            self.maskDir = pth
            self.initVars()
    
    def dumpRA2CSV(self, ra, name):
        with open('{}.csv'.format(name), "w",  newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                header = ['Session', '2D Area (mm^2)', '3D Area (mm^2)', 'Perimeter (mm)', 'Circularity', 'Compactness', 'Diameter (mm)', 'Mean Elevation']
                writer.writerow(header)
                
                for r in ra:
                    row = [r[7], r[0],  r[1], r[2],  r[3], r[4],  r[5], r[6]]
                    writer.writerow(row)
    
    def dumpLeaves2CSV(self, leaves, name):
        with open('{}.csv'.format(name), "w",  newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                header = ['Session', 'Leaf label', 'Blade 2D Area (mm^2)', 'Blade 3D Area (mm^2)', 
                'Mean Blade Elevation', 'Median Blade Elevation', 'Point Based Blade Elevation', 
                'Blade Length 2D (mm)', 'Blade Width 2D (mm)', 'Blade Length Point Based 2D (mm)', 
                'Blade Length Point Based 3D (mm)', 'Petiole Length 2D', 'Petiole Length 2D RP', 
                'Petiole Width 2D', 'Mean Petiole Elevation', 'Median Petiole Elevation', 'Point Based Petiole Elevation', 
                'Petiole Start Coord', 'Blade Insertion Point', 'Blade Tip Coord', 'Centroid', 'CentroidZ']
                writer.writerow(header)
                
                for s in leaves: 
                    for k,val in s[1].items():
                        row = [s[0], s[1][k].label, s[1][k].area2D, s[1][k].area3D, s[1][k].meanBladeInclination, s[1][k].medianBladeInclination, s[1][k].pointBasedBladeInclination,
                        s[1][k].bladeLength2D, s[1][k].bladeWidth2D, s[1][k].bladeLengthPointBased2D, s[1][k].bladeLengthPointBased3D, 
                        s[1][k].petioleLength2D, s[1][k].petioleLength2DRP, s[1][k].petioleWidth2D, s[1][k].meanPetioleInclination, 
                        s[1][k].medianPetioleInclination, s[1][k].pointBasedPetioleInclination, s[1][k].petioleStartCoord, 
                        s[1][k].bladeInsertionCoord, s[1][k].bladeTipCoord, s[1][k].centroid, s[1][k].centroidZ ]
                        writer.writerow(row)

    def calcDistBetween2Points(self, x0, y0, x1, y1):
        return math.hypot(x1-x0, y1-y0)
        
    def convert2IndexedLabels(self, labelMask, labelDict):
        palette = labelMask.getpalette()
        indexed = np.array(labelMask)
        
        num_colours = int(len(palette)/3)
        
        max_val = float(np.iinfo(indexed.dtype).max)
        
        map = np.array(palette).reshape(num_colours, 3) / max_val
        
        uLabels = np.unique(indexed)
        indexedNew = np.zeros(indexed.shape) - 1
        
        for u in uLabels:
            rgb = map[u] 
            if not tuple(rgb) in labelDict: 
                if u in list(labelDict.values()): 
                    maxLabel = np.max(list(labelDict.values())) + 1    
                    u = maxLabel
                
                labelDict.update({tuple(rgb) : u})
                if len(labelDict)>1:
                    c=np.bincount(np.array(list(labelDict.values())))
                    if np.any(c>1):
                        print('Duplicate label added: {}'.format(labelDict.values()))
                
            indexedNew[indexed == u]= labelDict[tuple(rgb)]
        
        blackLabel = labelDict[tuple(np.array([0.,0.,0.]))]
        
        if blackLabel > 0:
            indexedNew +=1 
            indexedNew[indexedNew == blackLabel+1] = 0

        return indexedNew.astype(np.uint8), labelDict
    
    def estimateRosetteArea(self, mask, Nz, camHeight=250, focalLength=25, cam=None):
        if cam is None:
            cam=ps.PGLens(focalLength)
        h,v=LensSensorPixelCalc(cam.focalLen, cam.sensorSize, camHeight, [2048, 2048]) 
        areaOfOnePixel=h*v
        howManyPixels=np.count_nonzero(mask)
        area2D=howManyPixels*areaOfOnePixel
        
        areaNz=(1/Nz)*areaOfOnePixel
        areaNz=np.multiply(areaNz, mask>0)
        areaNz=np.nansum(areaNz)
        
        return area2D, areaNz, h,v

    def getLengthAndElevation(self, startCoord, endCoord, sn, mask, px_h, type=LeafRegionProps.PETIOLE):
        if type == LeafRegionProps.PETIOLE:
            Nx = sn[:,:,0]
            Ny = sn[:,:,1]
            Nz = sn[:,:,2]
            snmask = (Nx==0) & (Ny==0) & (Nz == 0)
            Nx[snmask]=np.nan
            Ny[snmask]=np.nan
            Nz[snmask]=np.nan
            
            Nx[~mask]=np.nan
            Ny[~mask]=np.nan
            Nz[~mask]=np.nan
            az,elev,r=ps.cart2sph2(Nx, Ny, Nz)
            meanElev = np.nanmean(elev);
            d=self.calcDistBetween2Points(startCoord[0], startCoord[1], endCoord[0], endCoord[1])
            h = (d*px_h) / math.cos(math.radians(90)-meanElev)
            return 90-math.degrees(meanElev), h
        else:
            x = np.array([startCoord[0], endCoord[0]])
            y = np.array([startCoord[1], endCoord[1]])
            coefficients = np.polyfit(x, y, 1)
            polynomial = np.poly1d(coefficients)
            x_axis = np.linspace(x.min(),x.max(),(x.max())-(x.min()))
            y_axis = polynomial(x_axis)
            
            Nx = sn[y_axis.astype(np.int), x_axis.astype(np.int),0]
            Ny = sn[y_axis.astype(np.int), x_axis.astype(np.int),1]
            Nz = sn[y_axis.astype(np.int), x_axis.astype(np.int),2]
            
            az,elev,r=ps.cart2sph2(Nx, Ny, Nz)
            meanElev = np.nanmean(elev);
            d=self.calcDistBetween2Points(startCoord[0], startCoord[1], endCoord[0], endCoord[1])
            h = (d*px_h) / math.cos(math.radians(90)-meanElev)
            return 90-math.degrees(meanElev), h
    
    def estimateCentroidZ(self, petDist, petElev, bladeInsertionCoord, bladeElev, centroid, px_h):
        heightBIP=petDist * math.sin(math.radians(petElev))
        
        d = self.calcDistBetween2Points(centroid[0], centroid[1], bladeInsertionCoord[0], bladeInsertionCoord[1])  * px_h
        heightCentroidFromBIP = d * math.tan(math.radians(bladeElev))
        
        centroid_Z = heightBIP + heightCentroidFromBIP
        return centroid_Z
        
    def getStats(self, leaves, leafMasks, bladeMasks, stemMasks, labels, h, v):
        plantRegionProps ={};
        areaOfOnePixel = h*v
        plt.close('all')
        for i, l in enumerate(leaves):
            if not np.any(bladeMasks[i]):
                continue
            Nx = l[:,:,0]
            Ny = l[:,:,1]
            Nz = l[:,:,2]
            
            leafRP=LeafRegionProps();
            leafRP.label=labels[i]
            leafRP.h = h 
            leafRP.v = v 
        
            leafRP.area3D = np.nansum(bladeMasks[i] * (areaOfOnePixel * (1/Nz)))
            leafRP.area2D = np.sum(bladeMasks[i] * areaOfOnePixel)
            
            az,elev,r=ps.cart2sph2(Nx, Ny, Nz)
            
            STOREIMAGES = False
            if STOREIMAGES:
                leafRP.leafMask = leafMasks[i].copy()
                leafRP.elev = (90-np.degrees(elev.copy())) * leafRP.leafMask
            elev[bladeMasks[i]==False]=np.nan 
            leafRP.meanBladeInclination = 90-np.degrees(np.nanmean(elev))
            leafRP.medianBladeInclination = 90-np.degrees(np.nanmedian(elev))
            bladeRP = regionprops(bladeMasks[i].astype(np.int32)) 
            
            if len(bladeRP) != 1:
                print('Too many blades!');
            assert len(bladeRP) == 1
            
            leafRP.bladeWidth2D=bladeRP[0].minor_axis_length *h 
            leafRP.bladeLength2D=bladeRP[0].major_axis_length *v
            leafRP.centroid = bladeRP[0].centroid
            
            petioleMask = stemMasks[i].copy()
            petioleMask=remove_small_objects(petioleMask, 32)
            
            petioleRP = regionprops(petioleMask.astype(np.int32))
            
            # if visible:
            if petioleRP != []:
                leafRP.petioleWidth2D = petioleRP[0].minor_axis_length * h
                leafRP.petioleLength2DRP = petioleRP[0].major_axis_length * v
            else:
                leafRP.petioleWidth2D = 0
                leafRP.petioleLength2DRP = 0
            
            if np.any(petioleMask):
                leafRP.petioleMask = petioleMask.copy()
                
                petioleCoords=regionprops(label(petioleMask))[0].coords
                bladeCoords=bladeRP[0].coords
                Y = cdist(bladeCoords, petioleCoords, 'euclidean')
                matchLocations=np.where(Y==Y.min()) 
                coords=petioleCoords[matchLocations[1]]
                meanPetioleCoord=[np.mean(coords[:,0]), np.mean(coords[:,1])]
                coords=bladeCoords[matchLocations[0]]
                meanBladeCoord=[np.mean(coords[:,0]), np.mean(coords[:,1])]
                bladeInsertionCoord=[np.mean([meanPetioleCoord[0], meanBladeCoord[0]]), np.mean([meanPetioleCoord[1], meanBladeCoord[1]])]
                leafRP.bladeInsertionCoord=[bladeInsertionCoord[1], bladeInsertionCoord[0]] 
                
                az,elev,r=ps.cart2sph2(Nx, Ny, Nz)
                elev[petioleMask==False]=np.nan 
                leafRP.meanPetioleInclination = 90-np.degrees(np.nanmean(elev)) 
                leafRP.medianPetioleInclination = 90-np.degrees(np.nanmedian(elev))
                
                d=cdist(np.array([bladeInsertionCoord]), bladeCoords)
                matchLocations=np.where(d==d.max()) 
                bladeTipCoord = np.mean(bladeCoords[matchLocations[1]], axis=0)
                leafRP.bladeTipCoord = [bladeTipCoord[1], bladeTipCoord[0]] 
                leafRP.bladeLengthPointBased2D = self.calcDistBetween2Points(leafRP.bladeInsertionCoord[0], leafRP.bladeInsertionCoord[1], leafRP.bladeTipCoord[0], leafRP.bladeTipCoord[1])  * h
                bladeElev, bladeDist = self.getLengthAndElevation(np.array(leafRP.bladeInsertionCoord),np.array(leafRP.bladeTipCoord) , l, bladeMasks[i], h, type =LeafRegionProps.BLADE )
                leafRP.bladeLengthPointBased3D = bladeDist
                leafRP.pointBasedBladeInclination = bladeElev
                
                d=cdist(np.array([bladeInsertionCoord]), petioleCoords)
                matchLocations=np.where(d==d.max()) 
                petioleStartCoord = np.mean(petioleCoords[matchLocations[1]], axis=0)
                leafRP.petioleStartCoord = [petioleStartCoord[1], petioleStartCoord[0]] 
                leafRP.petioleLength2D = self.calcDistBetween2Points( leafRP.petioleStartCoord[0], leafRP.petioleStartCoord[1], leafRP.bladeInsertionCoord[0], leafRP.bladeInsertionCoord[1]) * h
                petElev, petDist = self.getLengthAndElevation(np.array(leafRP.petioleStartCoord), np.array(leafRP.bladeInsertionCoord), l, petioleMask, h,type =LeafRegionProps.PETIOLE )
                leafRP.petioleLength3D = petDist 
                leafRP.pointBasedPetioleInclination = petElev
                
                leafRP.centroidZ = self.estimateCentroidZ(petDist, petElev, leafRP.bladeInsertionCoord, bladeElev, leafRP.centroid, h)
                leafRP.bladeTipZ = self.estimateCentroidZ(petDist, petElev, leafRP.bladeInsertionCoord, bladeElev, leafRP.bladeTipCoord, h)
                
            plantRegionProps.update({leafRP.label : leafRP})
        
        return plantRegionProps
    
    def getLeafNormals(self, sn, labels):
        leaves = []
        bladeMasks = []
        stemMasks = [] 
        leafMasks = []
        maxLabel = labels.max()
        leafLabels = []
        
        for i in range(1, maxLabel+1): 
            leafMask = labels == i
            maskedLeaf=np.zeros(sn.shape)
            maskedLeaf[labels==i,:]=sn[labels==i,:]
            
            rp = regionprops(leafMask.astype(int))
            if rp != []:
                if rp[0].area < 1000:
                    disksize = 7
                if rp[0].area < 1500:
                    disksize = 9
                elif rp[0].area < 2400:
                    disksize = 11
                else:
                    disksize = 15
                    
                selem = disk(disksize) 
                bladeMask = binary_opening(leafMask, selem)
                
                bladeMasks.append(bladeMask.copy())
            
                stemMask = leafMask ^ bladeMask
                stemMasks.append(stemMask);
                
                if self.cbShowMasks.isChecked():
                    plt.close('all')
                    plt.figure()
                    plt.subplot(1,3,1)
                    plt.imshow(leafMask); 
                    plt.title('Leaf')
                    plt.subplot(1,3,2)
                    plt.imshow(bladeMask);
                    plt.title('Blade')
                    plt.subplot(1,3,3)
                    plt.imshow(stemMask);
                    plt.title('Stem');  plt.pause(1)
                
                leafLabels.append(i)
                
                leaves.append(maskedLeaf.copy())
                leafMasks.append(leafMask.copy())
            
        return leaves, leafMasks, bladeMasks, stemMasks, leafLabels
    
    def cvtFig2Numpy(self, fig):
    
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_argb(), dtype='uint8').reshape(height.astype(np.uint32), width.astype(np.uint32), 4)
        image=np.roll ( image, 3, axis = 2 )
        return image

    def dumpMaskLabelImage(self, session, plantNo, labelMask):
        fname = os.path.join(self.outputDir, '{}_{}.png'.format(plantNo, session))
        f = plt.figure()
        values = np.unique(labelMask.ravel())
        print("{}, {}".format(session, len(values)))
        im = plt.imshow(labelMask, cmap='jet');
      
        values = values[values>0]
        
        colors = [ im.cmap(im.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[i], label="Leaf {l}".format(l=values[i]) ) for i in range(len(values)) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.axis('off')
        #plt.show();
        labelIm = self.cvtFig2Numpy(f)
        plt.close(f)
        cv2.imwrite(fname, cv2.cvtColor(labelIm, cv2.COLOR_RGBA2BGRA));

    def process(self):
        now=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(self.outputDir):
            qm = QtGui.QMessageBox
            ret = qm.question(self,'Create Directory?', "Output results directory does not exist. Create?", qm.Yes | qm.No)
            
            if ret == qm.Yes:
                os.makedirs(self.outputDir)
        
        plantNos = [int(i) for i in self.edtPlants.text().split(',')]
        if not plantNos:
            qm.information(self,'Error', 'Please enter the plants to process separated by commas')
        startFolder = os.path.normpath(os.path.join(self.rootDir, str(self.startDir)))
        endFolder = os.path.normpath(os.path.join(self.rootDir, str(self.endDir)))
   
        d=glob.glob(os.path.join(self.rootDir, self.getLSType()))
        
        d.sort()
        d=[os.path.normpath(p) for p in d]
        
        if startFolder:
            startIdx =d.index(startFolder)
            endIdx =d.index(endFolder)
        
            d = d[startIdx:endIdx+1]
       
        total2Proc = len(d) * len(plantNos)
       
        progress = QProgressDialog("Generating results...", "Abort", 0, total2Proc, self);
        progress.setWindowTitle("Results Generator")
        progress.setWindowModality(QtCore.Qt.WindowModal);
        
        if self.cbOverride.isChecked():
            sensorSize = [float(s) for s in self.sensorSize.split('x')]
            lens=ps.Lens( int(self.focalLength), sensorSize)
            camHeight = float(self.camHeight) * 10
        else:
            psconfig = PSConfig.fromPropertiesFile(os.path.join(d[0], 'psconfig.properties'))
            lens = ps.PGLens(int(psconfig.lens)) 
            camHeight = psconfig.LS_VIS[0][2] * 10 
           
        imsz =  cv2.imread(os.path.join(d[0], 'im0.bmp'), cv2.IMREAD_GRAYSCALE).shape;
            
        h,v=LensSensorPixelCalc(lens.focalLen, lens.sensorSize, camHeight, imsz)
        
        flag = 0
        
        try:
            labelsDicts = {} 
            plantRegionProps = dict()
            for i, p in enumerate(plantNos):
                labels = dict() 
                if progress.wasCanceled():
                    break;
                raAll = [];
                leafStatsAll =[]
                for n,pth in enumerate(d):
                    progress.setValue(i*len(d) + n)
                    if progress.wasCanceled():
                        break;
                    
                    print('Processing Plant: {}, {}'.format(p, pth));
                    
                    A, z, snIm, shadowIm=loadPlant(pth, p, self.roiFile) 
                    session = os.path.basename(pth)
                    maskFname = os.path.join(self.maskDir, '{}_{}_mask.png'.format(p, session))
                    mask = cv2.imread(maskFname)
                    
                    if mask is None:
                        raise Exception('Mask file {} not found.'.format(maskFname))
                    
                    if len(np.unique(mask)) == 2:
                        mask=cv2.cvtColor( mask, cv2.COLOR_RGB2GRAY )
                        mask = mask.astype(bool)
                        flag = 1

                    else:
                        labelMask = cv2.imread(maskFname)
                        im = Image.fromarray(labelMask) 
                        labelMask = im.convert(mode="P", palette=Image.ADAPTIVE) 
                        
                        if p not in labelsDicts.keys():
                            labelsDicts[p] = {} 
                            
                        labelMask, labelsDicts[p] = self.convert2IndexedLabels(labelMask, labelsDicts[p])
                        
                        labels.update({p : labelMask})
                        self.dumpMaskLabelImage(session, p, labelMask) 
                        
                        leaves, leafMasks, bladeMasks, stemMasks, processedLeafLabels = self.getLeafNormals(snIm, labels[p])
                        plantLeavesStats = self.getStats(leaves, leafMasks, bladeMasks, stemMasks, processedLeafLabels, h, v)
                        leafStatsAll.append([os.path.basename(pth), plantLeavesStats])
                        
                        mask =cv2.cvtColor( mask, cv2.COLOR_RGB2GRAY )
                        mask = mask.astype(bool)
                    
                    Nx = snIm[:,:,0]
                    Ny = snIm[:,:,1]
                    Nz = snIm[:,:,2]
                    area2D, areaNz, h,v=self.estimateRosetteArea(mask, Nz, camHeight, lens.focalLen)
                    print('2D area: {:.2f}, 3D area: {:.2f}'.format(area2D, areaNz))
                    
                    if flag:
                        tmp = copy.deepcopy(mask)
                        tmp[mask == True] = 1
                        tmp[mask == False] = 0
                        tmp = tmp.astype(int)
                        props = regionprops(tmp)[0]
                        
                    else:
                        label_img = label(mask)
                        props = regionprops(label_img)[0] 
                    p_rw = props.perimeter * h
                    circ = (4 * math.pi * props.area) / (props.perimeter * props.perimeter) 
                    comp = props.solidity 
                    coords = find_contours(mask, .5)[0]
                    D = pdist(coords)
                    D = squareform(D)
                    N, [I_row, I_col] = np.nanmax(D)* h, np.unravel_index( np.argmax(D), D.shape )
                    
                    az,elev,r=ps.cart2sph2(Nx, Ny, Nz)
                    elev = (90-np.degrees(elev.copy()))
                    elev[mask==False] = np.nan
                    meanElev = np.nanmean(elev)
                    raAll.append([area2D, areaNz, p_rw, circ, comp, N, meanElev, os.path.basename(pth)])
               
                self.dumpRA2CSV(raAll, os.path.join(self.outputDir, '{}_{}_{}_{}_{}_rosette'.format(now, p, os.path.basename(self.rootDir), os.path.basename(startFolder), os.path.basename(endFolder))))
                if len(leafStatsAll):
                    self.dumpLeaves2CSV(leafStatsAll, os.path.join(self.outputDir, '{}_{}_{}_{}_{}_leaves'.format(now, p, os.path.basename(self.rootDir), os.path.basename(startFolder), os.path.basename(endFolder))))
            
            progress.setValue(total2Proc);
            qm = QtGui.QMessageBox
            qm.information(self,'Completed', 'Results can be found in {}.'.format(self.outputDir))
        except Exception as e:
            progress.cancel()
            qm = QtGui.QMessageBox
            qm.information(self,'Error', '{}'.format(e))
            raise e
        
    def closeEvent(self, event):
        self.saveToIniFile()
         
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = GenerateResultsGUI()
    window.show()
    window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    window.activateWindow()
    window.raise_()
    sys.exit(app.exec_())
 