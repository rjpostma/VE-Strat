# -*- coding: utf-8 -*-


#dependencies
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import compress

from skimage.morphology import disk
from skimage.segmentation import watershed,find_boundaries,mark_boundaries
from skimage.filters import rank
from skimage.feature import peak_local_max

from scipy import ndimage





###### FUNCTIONS ######

def identify_nucleus(img,op_br=3, pre_br=5, min_distance=25, save_path = None):
    
    #blur image for better segmentation
    img_normalized = cv2.blur(img, (pre_br,pre_br))
    
    #normalize image
    #substract background
    img_normalized = img_normalized - np.quantile(img_normalized,0.0001)
    #identify max and above
    rows, cols  = np.where(img_normalized > np.quantile(img_normalized,0.975))
    #set max
    img_normalized[rows, cols] = np.quantile(img_normalized,0.99)
    #normalize and set to CV_8UC1
    img_normalized = cv2.normalize(img_normalized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    #apply threshold
    _, thresh = cv2.threshold(img_normalized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #clean up thresholded image, using opening and closing
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(op_br,op_br)))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(op_br,op_br)))
    
    #fill holes
    thresh = ndimage.binary_fill_holes(thresh)
    
    ## Watershed ##
    #dist transform
    dist = ndimage.distance_transform_edt(thresh)
    #get local maxima
    localMax = peak_local_max(dist, min_distance=min_distance, labels=thresh)
    
    markersmax = np.zeros(dist.shape, dtype=bool)
    markersmax[tuple(localMax.T)] = True
    markersmax, _ = ndimage.label(markersmax)
    thresh_bool = thresh != 0
    #actual watershed
    labels = watershed(-dist, markersmax, mask=thresh_bool)
    
    #figure generation
    if(save_path != None):
        fig, ax = plt.subplots(3,2, figsize=(20, 30))
        fig.tight_layout()
        ax[0, 0].set_title('Original Image')
        ax[0, 0].imshow(img, cmap='gist_gray')
        ax[0, 0].axis("off")
        ax[0, 1].set_title('Normalized Image')
        ax[0, 1].imshow(img_normalized, cmap='gist_gray')
        ax[0, 1].axis("off")
        ax[1, 0].set_title('Thresholded Image')
        ax[1, 0].imshow(thresh)
        ax[1, 0].axis("off")
        ax[1, 1].set_title('Dist Transform')
        ax[1, 1].imshow(dist, cmap='viridis')
        ax[1, 1].axis("off")
        ax[2, 0].set_title('localMax')
        ax[2, 0].imshow(markersmax, cmap='prism')
        ax[2, 0].axis("off")
        ax[2, 1].set_title('Nuclei')
        ax[2, 1].imshow(labels, cmap='prism')
        ax[2, 1].axis("off")
        plt.savefig(save_path)

    return labels, markersmax, thresh, localMax



def identify_cells(img, markers, save_path = None, gradient_filter = 2):
    # normalize image
    rows, cols  = np.where(img > np.quantile(img,0.997))
    img_normalized = img.copy()
    img_normalized = img_normalized - np.quantile(img_normalized,0.001)
    img_normalized[rows, cols] = np.quantile(img_normalized,0.997)
    img_normalized = cv2.normalize(img_normalized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    # denoise image
    denoised = rank.median(img_normalized, disk(5))

    # local gradient
    gradient = rank.gradient(denoised, disk(gradient_filter))

    #seeded watershed, using the markers from the nuclei
    labels = watershed(gradient, markers)

    #overlay borders for display
    borders_overlay = mark_boundaries(img_normalized, labels)

    #identify borders
    borders = find_boundaries(labels)
    
    #convert borders to coordinate set
    rows, cols = np.where(borders==True)
    vor_points = [[i,j] for i,j in zip(rows,cols)]
    
    #figure generation
    if(save_path != None):
        fig, ax = plt.subplots(2,2, figsize=(20, 20))
        fig.tight_layout()
        ax[0, 0].set_title('Normalized Image')
        ax[0, 0].imshow(denoised, cmap='gist_gray')
        ax[0, 0].axis("off")
        ax[0, 1].set_title('Gradient Image')
        ax[0, 1].imshow(gradient, cmap='gist_gray')
        ax[0, 1].axis("off")
        ax[1, 0].set_title('Borders')
        ax[1, 0].imshow(borders_overlay)
        ax[1, 0].axis("off")
        ax[1, 1].set_title('Cells')
        ax[1, 1].imshow(labels, cmap='prism')
        ax[1, 1].axis("off")
        plt.savefig(save_path)
        
    return vor_points, img_normalized


def make_save_imageset(img, well, site, vor_points, path, kernel_size_radius, sample = 6000):
    n = 0
    assert len(vor_points) > 0
    while n < sample:
        x,y = random.choice(vor_points)
        if all([0 <= x-kernel_size_radius <= img.shape[0], 0 <= x+kernel_size_radius <= img.shape[0],
                0 <= y-kernel_size_radius <= img.shape[0], 0 <= y+kernel_size_radius <= img.shape[0]]):
            temp = img[x-kernel_size_radius:x+kernel_size_radius,y-kernel_size_radius:y+kernel_size_radius]
            cv2.imwrite(path + well + '_s' + str(site) + '_' + str(n) + '.png', temp)
            n = n+1


def make_store_imageset(img, vor_points, kernel_size_radius, sample = 6000):
    images = np.zeros((sample,kernel_size_radius*2,kernel_size_radius*2), dtype=int)
    n = 0
    coordinates = np.zeros((sample,2), dtype=int)
    while n < sample:
        x,y = random.choice(vor_points)
        if all([0 <= x-kernel_size_radius <= img.shape[0], 0 <= x+kernel_size_radius <= img.shape[0],
                0 <= y-kernel_size_radius <= img.shape[0], 0 <= y+kernel_size_radius <= img.shape[0]]):
            temp = img[x-kernel_size_radius:x+kernel_size_radius,y-kernel_size_radius:y+kernel_size_radius]
            images[n] = temp
            coordinates[n] = [x,y]
            n = n+1
    return images,coordinates


#################################### CLASSES  ###################################

class IX_multiplex:
    def __init__(self, nuclei, vecad):
        assert nuclei.shape == vecad.shape, "Images should match shape"
        self.nuclei = nuclei
        self.vecad = vecad
        self.maked_imageset = False
        
    def compute_nuclei(self,op_br,pre_br,min_distance,path=None):
        self.labels, self.markers, self.thresh, self.localMax = identify_nucleus(self.nuclei,op_br,pre_br,min_distance,path)
    
    def compute_cells(self, save_path = None):
        assert hasattr(self, 'markers'), "No nuclei present yet, run compute_nuclei first" 
        self.borders, self.vecad_normalized = identify_cells(self.vecad,self.markers,save_path)
    
    def generate_imageset_saved(self, path, well, site, kernel_size_radius, sample = 6000):
        assert hasattr(self, 'vecad_normalized'), "No segmented cells present" 
        self.maked_imageset = True
        self.saved_imageset_path = path
        make_save_imageset(self.vecad_normalized, well, site, self.borders, path, kernel_size_radius, sample)
    
    def assign_saved_imageset_path(self,path):
        self.maked_imageset = True
        self.saved_imageset_path = path
    
    def load_saved_imageset(self, img_size):
        IMG_ids = next(os.walk(self.saved_imageset_path))[2]
        IMGS  = np.empty((len(IMG_ids),img_size[0],img_size[1]))
        for j,i in enumerate(IMG_ids):
            temp = cv2.imread(self.saved_imageset_path+i,-1)
            IMGS[j] = temp
        self.vecad_ROI_imageset = IMGS
        
    def generate_imageset_stored_local(self, kernel_size_radius, sample = 6000):
        assert hasattr(self, 'vecad_normalized'), "No segmented cells present"
        self.vecad_ROI_imageset, self.vecad_ROI_coordinates = make_store_imageset(self.vecad_normalized, self.borders, kernel_size_radius, sample)

    def remove_imageset_stored(self):
        del self.vecad_ROI_imageset
        del self.vecad_ROI_coordinates
        print("Image Set Removed")
        
    def remove_nuclei_cells(self):
        #for garbage collection
        del self.labels
        del self.markers
        del self.thresh
        del self.localMax,
        del self.borders
        del self.vecad_normalized
        del self.nuclei
        del self.vecad
        
    def assign_embedded_imageset(self, embeddings, remove_imageset = True):
        assert hasattr(self, 'vecad_ROI_imageset') or self.maked_imageset, "No vecad_ROI_imageset present; set path to stored one or generate imageset"
        self.embeddings = embeddings
        if remove_imageset:
            self.remove_stored_imageset()
    
    def return_stored_imageset(self):
        print("returned imageset and coordinates")
        return self.vecad_ROI_imageset, self.vecad_ROI_coordinates

    def compute_embeddings_stored_imageset(self, encoder):
        self.encoded_imgs = encoder.predict(self.vecad_ROI_imageset)
    
    def return_embeddings(self):
        return self.encoded_imgs
        




class ImageXpress_site_container:
    def __init__(self, DNA_wavelength_path, VECAD_wavelength_path, site, well, plate):
        self.DNA_wavelength_path = DNA_wavelength_path
        self.VECAD_wavelength_path = VECAD_wavelength_path
        self.site = site
        self.well = well
        self.plate = plate
    
    def initiate_IX_multiplex(self):
        self.IX_multiplex = IX_multiplex(cv2.imread(self.DNA_wavelength_path,-1),cv2.imread(self.VECAD_wavelength_path,-1))
    
    def compute_nuc_cells_IX_multiplex(self, op_br, pre_br, min_distance, nuc_path = None, cell_path = None):
        self.IX_multiplex.compute_nuclei(op_br, pre_br, min_distance, nuc_path)
        self.IX_multiplex.compute_cells(cell_path)
    
    def compute_imageset_and_embed(self, kernel_diameter, encoder, samples = 6000):
        self.IX_multiplex.generate_imageset_stored_local(kernel_diameter, samples)
        self.IX_multiplex.remove_nuclei_cells
        self.IX_multiplex.compute_embeddings_stored_imageset(encoder)
        
    def compute_imageset_store(self, path, kernel_diameter, samples = 6000):
        self.IX_multiplex.generate_imageset_saved(path, self.well, self.site, kernel_diameter, samples)
    
    def return_embeddings(self):
        return self.IX_multiplex.return_embeddings()

    def return_info(self):
        return self.plate, self.well, str(self.site)




class ImageXpress_filetree:
    def __init__(self, path, DNA_wavelength, VECAD_wavelength):
        assert type(DNA_wavelength) == int, "DNA_wavelength should be dtype int"
        assert type(VECAD_wavelength) == int, "VECAD_wavelength should be dtype int"
        self.DNA_wavelength = DNA_wavelength
        self.VECAD_wavelength = VECAD_wavelength
        self.path = path
        self.plate = 'unknown'
        self.sites_number = 1
        self.wavelength_number = 2
        self.well_number = 1

    def images_list(self):
        # files in directory
        IMG_ids = next(os.walk(self.path))[2]
        # get tifs
        IMG_ids = [s for s in IMG_ids if any(xs in s for xs in ['.tif','.TIF','.tiff','.TIFF'])]
        # remove thumbnales
        matching = [s for s in IMG_ids if any(xs in s for xs in ['humb','THUMB'])]
        IMG_ids = [e for e in IMG_ids if e not in matching]
        self.IMG_ids = IMG_ids
        
        well = np.zeros((len(IMG_ids)),dtype='<U6')
        position = np.zeros((len(IMG_ids)),dtype=int)
        channel = np.zeros((len(IMG_ids)),dtype=int)
        
        #the main structure of the program is a simple for loop, getting all the images processed
        for n, id_ in tqdm(enumerate(IMG_ids), total=len(IMG_ids)):
            #Append measurements to numpy array 
            plate = str(id_.split("_")[0])
            well[n] = str(id_.split("_")[1])
            position[n] = id_.split("_")[2][1:]
            channel[n] = id_.split("_")[3][1]

        self.plate = plate
        self.sites_number = np.max(position)
        self.wavelength_number = np.max(channel)
        self.well_number = len(np.unique(well))
        
        
        self.stacks = []
        for i in np.unique(well):
            for j in range(self.sites_number):
                img_dna = list((well == i) & (position == j) & (channel == self.DNA_wavelength))
                img_vecad = list((well == i) & (position == j) & (channel == self.VECAD_wavelength))
                if np.sum(img_dna) == 1 & np.sum(img_vecad) == 1:
                    path_dna = list(compress(IMG_ids, img_dna))[0]
                    path_vecad = list(compress(IMG_ids, img_vecad))[0]
                    self.stacks = np.append(self.stacks,ImageXpress_site_container(self.path + path_dna,
                                        self.path + path_vecad, j, str(i), self.plate))
    
    def sites_initiate_IX_multiplex(self):
        for i in tqdm(range(len(self.stacks))):
            self.stacks[i].initiate_IX_multiplex()

    def sites_IX_multiplex_saveandgenerate_imageset(self, store_path, kernel_diameter, samples):
        for i in tqdm(range(len(self.stacks))):
            self.stacks[i].compute_nuc_cells_IX_multiplex(3,5,25)
            self.stacks[i].compute_imageset_store(store_path, kernel_diameter, samples)
    
    def sites_IX_multiplex_localstore_and_generate_imageset(self, kernel_diameter, encoder, samples):
        for i in tqdm(range(len(self.stacks))):
            self.stacks[i].compute_nuc_cells_IX_multiplex(3,5,25)
            self.stacks[i].compute_imageset_and_embed(kernel_diameter, encoder, samples)
    
    def return_sites(self):
        return self.stacks







