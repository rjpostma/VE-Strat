# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:29:25 2024

@author: Rudmer Postma
"""

import os
#change working directory to where your files are
os.chdir("")
print(os.getcwd())

#gettin all the functions and dependencies from the other file
from cell_border_identification_function_classes import *


#dependencies for the model
import tensorflow
from keras.models import Model 
from keras import layers
from keras.models import load_model




#%%

#########  THESE VARIABLES MUST BE SET  ########

""" 
The pipeline we use is highly tuned to use with the ImageXpress microscope.
It outputs 16 bits images per channel with filename plate_well_site_channel
So we extract that info as metadata from the filename, but we need to specify 
first which channels correspond to the VE-Cadherin, and Nuclei for the process to go automatically
"""


NUCLEI_w = 1
VECAD_w = 3

#######
## The size of the ROI is determined by this radius, do 32 corresponds to a ROI of 64x64
kernel_size_radius = 32

# Total amounts of ROIs extracted from each image.
sampling_total = 6000


#%%
#loading the model

encoder = load_model("model-encoder-trainingset.keras")
encoder.summary()



#%%

"""
Basically the procedure is as follows, we specify where the ILLUMINATION CORRECTED images are,
and then we scan the directory and images are automatically sorted for well, site, and channel,
and then generate the ROIs and embeddings all at once.

For the liver example in the paper, this is done by some simple functions.
But even the ecoding can be done on a collection of ROIs stored already using some of the other functions.

"""


plate1 = ImageXpress_filetree("H:/liver_set/plate1/",NUCLEI_w,VECAD_w)
plate1.images_list()
plate1.sites_initiate_IX_multiplex()
plate1.sites_IX_multiplex_localstore_and_generate_imageset(kernel_size_radius,encoder,sampling_total)


plate2 = ImageXpress_filetree("H:/liver_set/plate2/",NUCLEI_w,VECAD_w)
plate2.images_list()
plate2.sites_initiate_IX_multiplex()
plate2.sites_IX_multiplex_localstore_and_generate_imageset(kernel_size_radius,encoder,sampling_total)



#%%

"""
For the examples we used, we used a UMAP calculated on these embeddings
Subsequently, we use that UMAP and clusters to transform the other datasets
"""


encoded_imgs_all = np.load("encoder_umap.npy")
scaler = RobustScaler().fit(encoded_imgs_all)
reducer = umap.UMAP(n_neighbors=5,min_dist=0,n_components=2,random_state=3, metric='cosine')
X_umap = reducer.fit_transform(scaler.transform(encoded_imgs_all))

hdbscan_clusters = hdbscan.HDBSCAN(
    min_samples=5,
    min_cluster_size=50,prediction_data = True
).fit(X_umap)


## Other data was transformed using the following functions, like so, or used in a for loop to score for each image
X_umap = reducer.transform(scaler.transform(plate2.stack[1].return_embeddings))
labels, strengths = hdbscan.approximate_predict(hdbscan_clusters ,X_umap)


