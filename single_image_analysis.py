# -*- coding: utf-8 -*-

import os
#change working directory to where this file is
os.chdir("H:/LUMC/VEcadDeepLearning/GITHUB/") #please set your wd so the cell_border_identification_function_classes script can be found
print(os.getcwd())

#gettin all the functions and dependencies from the other file
from cell_border_identification_function_classes import *

#dependencies for the model
import tensorflow
import keras
from keras.models import Model 
from keras import layers
from keras.models import load_model
import cv2
from sklearn.preprocessing import StandardScaler, RobustScaler
import umap #form the package UMAP-learn
import hdbscan
import seaborn as sns
import pickle



#%%
""" 
The pipeline we use is highly tuned to use with the ImageXpress microscope.
But it is also usable with single images. For that we use the IX_multiplex class to process the images
"""


#########  THESE VARIABLES MUST BE SET  ########
#######
## The size of the ROI is determined by this radius, so 32 corresponds to a ROI of 64x64
kernel_size_radius = 32

# Total amounts of ROIs extracted from each image.
sampling_total = 6000

#loading the model
encoder = load_model("model-encoder.keras")



#%%
"""
Basically the procedure is as follows, we load the the ILLUMINATION CORRECTED border and nuclei images
Then we just calculate the embeddings and extract them.
"""

#reading original images
nuclei = cv2.imread("./Single images/TNF-nucl.tif",-1)
borders = cv2.imread("./Single images/TNF-borders.tif",-1)

#load the images into the IX_multiplex class

stack = IX_multiplex(nuclei, borders)
stack.compute_nuclei(3, 5, 25) #using the 3 pixel opening and closing for nuclei cleanup, 5 pixel smoothing brush, and 25 pixel minimal distance
stack.compute_cells() #traces the cell borders
stack.generate_ROIs(kernel_size_radius, sampling_total) #gets the ROI images from the original images
stack.compute_embeddings(encoder) #using the encoder defined above to calculate the embeddings





#%%
"""
For the examples we used, we used a UMAP calculated on a subset of embeddings
Subsequently, we use that UMAP and clustering to transform the all other datasets
"""
#loading the UMAP, scaler, and clustering
hdbscan_clusters = pickle.load(open("hdbscan_clusters.pkl", "rb"))
umap_reducer = pickle.load(open("umap_reducer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# calculating the clustering and umap
X_umap, labels = compute_umap_clusters(umap_reducer,scaler,hdbscan_clusters, stack.return_embeddings())

# displaying the umap
display_umap(X_umap, labels,legend=True)



