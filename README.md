# VAMPIRE (Visually Aided Morpho-Phenotyping Image Recognition)
**A robust method to quantify cell morphological heterogeneity**

**1. System requirements**\
    OS : Windows 10 (64 bit) Version 1909\
    Software is not compatible with older versions of Windows.\
    Mac OS is not officially supported, but it may work.\
    Non-standard hardware is not required.
    
**2. Installation Guide**\
    **Executable file option:**\
    No installation required. Download the executable file from https://github.com/kukionfr/VAMPIRE_open/releases/download/executable/vampire.exe \
    Open the executable file to launch the graphic user interface (GUI) of the software\
    Executable file "vampire.exe" can be also found in Supplementary Data 2 of manuscript.\
**PIP installation option:**\
    Type the following into command prompt window to install vampireanlysis on PYPI (the Python package index) using pip installer
    
    pip install vampireanalysis
    
To launch the GUI, type "vampire" into command prompt window.
    
**3. Demo**\
    Instructions to run on data can be found in the Procedure section of the manuscript.\
    Expected output of the procedure is provided in the Figure 5 of the manuscript and also in the supplementary files.\
    Expected run time for demo :\
        Step 1-2, Segment cells or nuclei, 5~10 mins\
        Step 3, Create a list of images to build the shape-analysis model, 1-3 mins\
        Steps 4-9, Build shape-analysis model in VAMPIRE, 1-5 mins\
        Steps 10-12, Application of the model to analyze shapes across conditions, 1-5 mins\
        Total, steps 1-12, complete VAMPIRE analysis, 8-23 mins
        
**4. Instructions for use**\
    Instructions to run on data can be found in the Procedure section of the manuscript.\
    By following the Procedure section, the users can reproduce the expected output data provided in the supplementary files.

**5. Code functionality**\
    The source code can be installed using pip: “pip install vampireanalysis” for Python 3.7 or later.\
    After installation using pip, type “vampire” in the command window prompt to launch the GUI.\
    
    
•	vampire.py : launch Tk interface for VAMPIRE GUI.\
•	mainbody.py : read the boundaries of cells or nuclei and process them through three key functions of VAMPIRE analysis: 1. Registration 2. PCA 3. Cluster.\
•	collect_selected_bstack.py : read the boundaries of cells or nuclei based on the CSV files that contains list of image sets to build or apply the VAMPIRE model.\
•	bdreg.py: register boundaries of cells or nuclei to eliminate rotational variance.\
•	pca_bdreg.py : apply PCA to the registered boundaries.\
•	PCA_custom.py  : principal component analysis code.\
•	clusterSM.py : apply K-means clustering to PCA processed boundaries of cells or nuclei and assign the cluster number label to each cell or nuclei.\
•	update_csv.py : generate VAMPIRE datasheet based on the assigned cluster label.\
Codes that are not mentions here belongs to the codes explained. The provided explanation applies to those as well.
