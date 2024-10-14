# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:29:25 2024

@author: Rudmer Postma
"""

import os
#change working directory to where this file is
os.chdir("H:/GITHUB/VESTAT/")
print(os.getcwd())

#gettin all the functions and dependencies from the other file
from cell_border_identification_function_classes import *

#dependencies for the model
import tensorflow
import keras
from keras.models import Model 
from keras import layers
from keras.models import load_model

from sklearn.preprocessing import StandardScaler, RobustScaler
import umap #THe package UMAP-learn
import hdbscan
import seaborn as sns
import pickle



#%%

#########  THESE VARIABLES MUST BE SET  ########

""" 
The pipeline we use is highly tuned to use with the ImageXpress microscope.
It outputs 16 bits images per channel with filename plate_well_site_channel
So we extract that info as metadata from the filename, but we need to specify 
first which channels correspond to the VE-Cadherin, and Nuclei for the process to go automatically
"""

# setting the channel corresponding to the nuclei or vecadherin.
NUCLEI_w = 4
VECAD_w = 2

#######
## The size of the ROI is determined by this radius, so 32 corresponds to a ROI of 64x64
kernel_size_radius = 32

# Total amounts of ROIs extracted from each image.
sampling_total = 2000


#%%
#loading the model
encoder = load_model("model-encoder.keras")



#%%
"""
Basically the procedure is as follows, we specify where the ILLUMINATION CORRECTED images are,
and then we scan the directory and images are automatically sorted for well, site, and channel,
and then generate the ROIs and embeddings all at once.
"""

control = ImageXpress_filetree("./Examples/Control/",NUCLEI_w,VECAD_w)
control.index_images()
control.load_images_into_IX_multiplex()
control.IX_multiplex_compute_embeddings(kernel_size_radius,encoder,sampling_total)
wells,embeddings = control.return_embeddings_well()


#%%
"""
For the examples we used, we used a UMAP calculated on a subset of the all embeddings present in the dataset
Subsequently, we use that UMAP and clusters to transform the other datasets
"""

#   loading the scaler, umap, and clusteringalgorithm
hdbscan_clusters = pickle.load(open("hdbscan_clusters.pkl", "rb"))
umap_reducer = pickle.load(open("umap_reducer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# them we just compute the clustering and display the UMAP
X_umap, cluster = compute_umap_clusters(umap_reducer,scaler,hdbscan_clusters, embeddings[0])
display_umap(X_umap, cluster,legend=True)


