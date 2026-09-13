# Depth based human action classifier using Microsoft's Kinect v1, PoC

> **DISCLAIMER:** This is a proof-of-concept and was made for learning purposes, the models can suffer from overfitting, and the scripts/openni-python3 package are not very polished and are prone to bugs.

## Citations:
*Human Activity Recognition Process Using 3-D Posture Data. S. Gaglio, G. Lo Re, M. Morana. In IEEE Transactions on Human-Machine Systems. 2014 doi: 10.1109/THMS.2014.2377111*

*Liu, Mengyuan & Liu, Hong & Hu, Qianshuo & Ren, Bin & Yuan, Junsong & Lin, Jiaying & Wen, Jiajun. (2025). 3D Skeleton-Based Action Recognition: A Review. 10.48550/arXiv.2506.00915. *

*Yuan, Lin & He, Zhen & Wang, Qiang & Xu, Leiyang & Ma, Xiang. (2023). Improving Small-Scale Human Action Recognition Performance Using a 3D Heatmap Volume. Sensors. 23. 10.3390/s23146364. *

Realtime SVM/RF-based action recognition and classification using a modified and augmented KARD dataset, libfreenect and OpenNI2 backend for device interaction and skeletal tracking respectively

## Demo
https://raw.githubusercontent.com/MerGr/Kinectv1-human-action-classifier/refs/heads/main/assets/demo.mp4

## Requirements
–	Python 3.8+ with pip
–	Kinect V1 (aka Xbox 360 Kinect) sensor (Note for Windows: must use Zadig LibusbK driver for all 3 Xbox NUI devices)
–	libfreenectv1 latest commit (0.7.5 as of 2026)
–	latest OpenNI2 + PrimeSense NiTE2 binaries (2.2.x)
-   Cmake Build Tools for compiling libfreenect and openni-python3 python package

## Setup

First, clone this repository using the command

`git clone --recurse-submodules https://github.com/MerGr/Kinectv1-human-action-classifier.git`

then, compile libfreenect with these compile flags

`-DBUILD_PYTHON3=OFF -DBUILD_OPENNI2_DRIVER=ON -DBUILD_REDIST_PACKAGE=OFF`

> Windows might require explicitly specifying win32-pthreads and libusb /include and /lib folders

in the project root directory, create a Python 3 virtual environment, activate it and compile openni-python3 as specified in it's own README.MD, and then run :

`pip install -r requirements.txt`

finally, copy (or symlink) the NiTE2 folder (commonly found in NiTE2/Redist/NiTE2) containing \*.dat files (proprietary blobs used for skeletal tracking) to the working directory of the scripts that would call for OpenNI2, mainly `webui` and `scripts`

## Usage

Connect your Kinect to your PC, and then in `webui` folder, run:

`streamlit run app.py`

the app defaults to using the Random Forest model. You can change the settings inside the python scripts

## Notebooks

the jupyter notebook stored in `notebooks` will:
- Download the KARD dataset from the University of Monderley and the extra activity (a11) captured via `scripts/capture_data.py` from my own website (create an issue if the files is no longer hosted, or create your own dataset conformant to the way KARD was built via the same scripts)
- Prepares, processes and augments the dataset, splits into training and test data
- trains a snippet of the training data using GridSearchCV using an explicitly selected grid of parameters
- trains with the entire training dataset with the best parameters found, evaluates and then exports to joblib files to be used by the webapp
