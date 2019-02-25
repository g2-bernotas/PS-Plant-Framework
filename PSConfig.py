import configparser
import json
import os
'''
Class to load and store PSConfig.properties values
'''
filename='PSConfig.properties'
class PSConfig:
    def __init__(self,LS_VIS, LS_NIR, LS_VISNIR, toBeProcessedDir, archivePath, COMPort, x1, x2, y1, y2, showResults, export3DData, environment, lens, applyLightingCompensation, crop2Plant, subtractAmbient):
        self.LS_VIS=LS_VIS
        self.LS_NIR=LS_NIR
        self.LS_VISNIR=LS_VISNIR
        self.toBeProcessedDir=toBeProcessedDir 
        self.archivePath=archivePath 
        self.COMPort=COMPort    
        self.roi=[x1, x2, y1, y2] 
        self.showResults=showResults
        self.export3DData=export3DData 
        self.environment=environment 
        self.lens=lens 
        self.applyLightingCompensation=applyLightingCompensation 
        self.crop2Plant= crop2Plant 
        self.subtractAmbient= subtractAmbient 
    
    def overwriteWithGuiProps(self, guiConfig):
        self.applyLightingCompensation=guiConfig.applyLightingCompensation 
        self.crop2Plant= guiConfig.crop2Plant 
        self.subtractAmbient= guiConfig.subtractAmbient 
        self.roi=guiConfig.roi
        
    @classmethod
    def fromPropertiesFile(cls, filename):
        config = configparser.ConfigParser();
        config.read(filename);
        
        LS_VIS=json.loads(config['LightSourceVectors']['LS_VIS'])
        LS_NIR=json.loads(config['LightSourceVectors']['LS_NIR'])
        LS_VISNIR=LS_VIS + LS_NIR
        
        toBeProcessedDir=config['Directories']['toBeProcessedPath']
        archivePath=config['Directories']['archivePath']
        
        COMPort=config['Arduino']['COMPort']
        
        x1=json.loads(config['ROI']['x1'])
        x2=json.loads(config['ROI']['x2'])
        y1=json.loads(config['ROI']['y1'])
        y2=json.loads(config['ROI']['y2'])
        
        showResults=json.loads(config['Misc']['showResults'])
        export3DData=json.loads(config['Misc']['export3DData'])
        
        environment='PC' 
        try:
            environment=config['Misc']['environment']
        except KeyError as k:
            environment='PC' # Default    
        
        lens=16 # the focal length of the lens
        try:
            lens=json.loads(config['Misc']['lens'])
        except KeyError as k:
            lens=16 # Default  
            
        applyLightingCompensation=1
        try:
            applyLightingCompensation=json.loads(config['Misc']['applyLightingCompensation'])
        except KeyError as k:
            applyLightingCompensation=1 # Default    
        
        crop2Plant=0
        try:
            crop2Plant=json.loads(config['Misc']['crop2Plant'])
        except KeyError as k:
            crop2Plant=0  # Default    
        
        subtractAmbient=0
        try:
            subtractAmbient=json.loads(config['Misc']['subtractAmbient'])
        except KeyError as k:
            subtractAmbient=0  # Default    
        
        return cls(LS_VIS, LS_NIR, LS_VISNIR, toBeProcessedDir, archivePath, COMPort, x1, x2, y1, y2, showResults, export3DData, environment, lens, applyLightingCompensation, crop2Plant, subtractAmbient)
        
    def saveProps(self,pth):
        cfgFile=open(os.path.join(pth, filename),'w')
        config = configparser.ConfigParser();
        
        config.add_section('LightSourceVectors')
        config.set('LightSourceVectors','LS_VIS', '[' + ', '.join(map(str, self.LS_VIS)) + ']')
        config.set('LightSourceVectors','LS_NIR', '[' + ', '.join(map(str, self.LS_NIR)) + ']')
        
        config.add_section('Directories')
        config.set('Directories','toBeProcessedPath', self.toBeProcessedDir)
        config.set('Directories','archivePath', self.archivePath)
        
        config.add_section('Arduino')
        config.set('Arduino','COMPort', self.COMPort)
        
        config.add_section('ROI')
        config.set('ROI','x1', str(self.roi[0]))
        config.set('ROI','x2', str(self.roi[1]))
        config.set('ROI','y1', str(self.roi[2]))
        config.set('ROI','y2', str(self.roi[3]))
        
        config.add_section('Misc')
        config.set('Misc','showResults', str(self.showResults))
        config.set('Misc','export3DData', str(self.export3DData))
        config.set('Misc','environment', self.environment)
        config.set('Misc','lens', str(self.lens))
        config.set('Misc','applyLightingCompensation', str(self.applyLightingCompensation))
        config.set('Misc','crop2Plant', str(self.crop2Plant))
        config.set('Misc','subtractAmbient', str(self.subtractAmbient))
       
        config.write(cfgFile)
        cfgFile.close()
    # Provides storage for those options on the GUI
    @classmethod
    def fromGUI(cls, applyLightingCompensation, crop2Plant, subtractAmbient, roi):
        return cls([],[],[], '', '', '', roi[0], roi[1], roi[2],roi[3],0,0,'', 0, int(applyLightingCompensation), int(crop2Plant), int(subtractAmbient));
    
    def writeConfig(self, pth):
        with open(os.path.join(pth,'PSConfig.properties'), 'w') as configfile:    # save
            config.write(configfile)
        
if __name__ == '__main__':
    o=PSConfig.fromPropertiesFile('PSConfig.properties')
    o.saveProps('')
    print(o.LS_VISNIR)