## PS-Plant data processing software 
For "A photometric stereo-based 3D imaging system using computer vision and deep learning for tracking plant growth" paper.

This is an implementation of PS-Plant software in Python 3.5. The software has six components that is the backbone for processing data acquired using PS-Plant plant phenotyping system. Each component runs as a GUI enabling the user to interact with the software. The components are named as follows:

1. Generate Adaptive Light Source Vectors (GenerateAdaptiveLSVGUI)
2. Generate SNZShadowImAndAlbedo_adaptiveLS.npz (BatchProcessRawSessionsGUI)
3. Generate Rosette Masks (MaskGenGUI)
4. Generate Leaf Masks (LeafSegmentationGUI)
5. Generate Tracked Leaf Masks (TrackingGUI)
6. Generate Results (GenerateResultsGUI)

## Dependencies
* numpy==1.15.4
* opencv-python==3.2.0.8
* matplotlib==2.2.0
* retrying==1.3.3
* scikit-image==0.14.1
* scipy==1.1.0
* pandas==0.23.4
* trackpy==0.4.1
* tensorflow-gpu==1.9.0
* keras==2.2.0
* visvis==1.10.0
* imgaug==0.2.6
* pyqt4==4.11.4 (https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4)
* PyCapture==2.11.425 (https://www.ptgrey.com/support/downloads)
* IPython
* FlyCapture 2.13.3.31 SDK (https://www.ptgrey.com/support/downloads

The repository includes:
* Source code for PS-Plant data analysis using GUI
* Modified Mask R-CNN code for Arabidopsis leaf segmentation (adapted from Matterport (https://github.com/matterport/Mask_RCNN), original Mask R-CNN paper by He et al. (2017))
* Pre-trained weights for leaf segmentation using grayscale image modality and Mask R-CNN (also available from: https://liveuweac-my.sharepoint.com/:f:/g/personal/gytis2_bernotas_live_uwe_ac_uk/EqEuMisdwI5CjVjnfB7rofwBlFWJaAZTrEeuSQUjgSGHjw?e=Av62i5)
* Dataset of manually annotated Arabidopsis leaves for training the Mask R-CNN models (also available from: https://datashare.is.ed.ac.uk/handle/10283/3200)
* PS-Plant data acquisitions (10) of Arabidopsis plants for testing purposes (also available from: https://liveuweac-my.sharepoint.com/:f:/g/personal/gytis2_bernotas_live_uwe_ac_uk/EqEuMisdwI5CjVjnfB7rofwBlFWJaAZTrEeuSQUjgSGHjw?e=Av62i5)
* Results expected after running the software with the provided data
* Requirements.txt file listing the required packages for successfully running the software


# Starting PS-Plant data
The easiest way to investigate the PS data is to run 'psgui3.py' GUI where the GUI displays integrated 3D surface using Frankot and Chellappa (1988), surface normal directions in x, y and z directions  and albedo image. Click on 'Process from archive' and navigate to the provided raw PS data acquisitions in TestData/RawData.

# GenerateAdaptiveLSVGUI 
This script displays a GUI to generate adaptive light source vectors that will be used to generate more accurate 3D representations using PS.
	
In the GUI you have to enter:
	- Focal length of the camera lens;
	- Sensor size of the camera;
	- Resolution of the camera;
	- Path to the PSConfig.properties file;
	- Path where the adaptive light source file will be stored.
	
Generated light source vectors are also available from here: https://liveuweac-my.sharepoint.com/:f:/g/personal/gytis2_bernotas_live_uwe_ac_uk/EqEuMisdwI5CjVjnfB7rofwBlFWJaAZTrEeuSQUjgSGHjw?e=Av62i5

# BatchProcessRawSessionsGUI
This is the GUI for processing raw files and generating PS outputs that are stored in SNZShadowImAndAlbedo_adaptiveLS.npz files. 

In the GUI user has to enter:
	- Data directory and light source type (VIS, NIR or VISNIR);
	- Start and end session names;
	- Whether to use or not the adaptive light source vectors (if yes, 
	provide the path to the adaptive light source vectors).
	
The user may run the script on the provided PS-Plant data acquisitions, however, the SNZShadowImAndAlbedo_adaptiveLS.npz files are already provided in the directories in TestData/RawData directory.

# MaskGenGUI
This script displays a GUI to generate rosette masks for the chosen PS-Plant acquisition sessions and desired region of interests (ROI). The GUI allows user to select parameters for better segmentation (threshold, min area, filter size).
	
In the GUI you have to enter:
	- Path to the PS data;
	- Start and end session names from drop down list;
	- ROI file;
	- Plant numbers of interest;
	- Path to the output directory;
	- The user may also preview the output of the chosen parameters (threshold, 
	min area, filter size).
	
Example roi.txt is provided in TestData for the directories in TestData/RawData directories with NIR suffix. The results are provided in TestData/Results/RosetteMasks.
	
# LeafSegmentationGUI
This is the GUI for generating individual leaf segmentations. 

In the GUI, user has to enter:
	- Cropped image data directory (provided model is for images size of 512x512)
	- Save directory
	- Path to the pre-trained Mask R-CNN model weights (model trained on grayscale 
	images can be donwloaded from: 
	https://liveuweac-my.sharepoint.com/:f:/g/personal/gytis2_bernotas_live_uwe_ac_uk/EqEuMisdwI5CjVjnfB7rofwBlFWJaAZTrEeuSQUjgSGHjw?e=yYT8TR
	
Raw data is provided in TestData/RawData/Cropped directory. Example roiLeaf.txt is provided in TestData for the directories in TestData/RawData directories with NIR suffix. The results are provided in TestData/Results/RosetteMasks.

The results are provided in TestData/Results/LeafMasks.

# TrackingGUI
This is the GUI for tracking leaf movements. 

In the GUI you have to enter:
	- Path to the untracked masks;
	- Path to the output directory;
	- Parameters:
		- Span (defaut 0)
		- Memory (default 20)
		- Max displacement (default 30)
		- Min frames (default 10)

Raw data is provided in TestData/RawData/LeafMasks directory or the user generated images from 'LeafSegmentationGUI'. The results are provided in TestData/Results/TrackedLeafMasks.

#
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
	
Raw data is provided in TestData/RawData/TrackedLeafMasks and TestData/RawData/RosetteMasks directories or the user generated images from 'MaskGenGUI' or 'TrackingGUI'. The results are provided in TestData/Results/DataExtraction for both rosette and leaf segmentations.