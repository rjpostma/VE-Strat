# -*- coding: utf-8 -*-




# Dependencies
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

## other dependencies
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler
import umap #From the package UMAP-learn!!
import hdbscan
import seaborn as sns
import pickle



#%%

#########  THESE VARIABLES MUST BE SET  ########
""" 
The pipeline we use is highly tuned to use with the ImageXpress microscope.
It outputs 16 bits images per channel with filename plate_well_site_channel
So we extract that info as metadata from the filename, but we need to specify 
which channels correspond to the VE-Cadherin and Nuclei for the process
"""

# setting the channel corresponding to the nuclei or vecadherin.
NUCLEI_w = 1
VECAD_w = 3

#######
## The size of the ROI is determined by this radius, so 32 corresponds to a ROI of 64x64
kernel_size_radius = 32

# Total amounts of ROIs extracted from each image.
sampling_total = 1000


#%%
#loading the model
encoder = load_model("model-encoder.keras")



#%%
"""
Basically the procedure is as follows, we specify where the ILLUMINATION CORRECTED images are,
and then we scan the directory and images are automatically sorted for well, site, and channel,
and then generate the ROIs and embeddings all at once.
"""

liver = ImageXpress_filetree("./Examples/Liver",NUCLEI_w,VECAD_w)
liver.index_images()
liver.load_images_into_IX_multiplex()
liver.IX_multiplex_compute_embeddings(kernel_size_radius,encoder,sampling_total)

wells,embeddings = liver.return_embeddings_well()

# Load the UMAP model, the scalar for the model, and the hdbscan clustering from file
hdbscan_clusters = pickle.load(open("hdbscan_clusters.pkl", "rb"))
umap_reducer = pickle.load(open("umap_reducer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# returning clustercount from the embeddings
clustercount = compute_cluster_occupancy(embeddings,umap_reducer,scaler,hdbscan_clusters)
# specific for this example we extract the patient group from the images
y_group = [str(i)[0] for i in wells]




#%%
# Making a PCA plot of the resulting profiles
pca = PCA(n_components=10, random_state=0)
X_pca = pca.fit_transform(StandardScaler().fit_transform(clustercount))


sns.set_theme(style="whitegrid")
plt.figure(figsize=(10,10))
sns.scatterplot(x=X_pca[:,0], 
                y=X_pca[:,1],
                     hue= y_group,
                     style = y_group,
                     alpha=.8, palette="muted"
                     )
plt.title('PCA plot')
plt.xlabel("PC 1")
plt.ylabel("PC 2")









