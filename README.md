# VE-Stat
Deep learning tool for VE-Cadherin cell border quantification and stratification

Training images for the model are included in the TRAINING folder.
Examples of different stimuli are also included in the TRAINING folder

The pipeline we created is mostly geared towards the ImageXpress microscope filestructure, but can be adapted to other image sets as well

Examples of how we implement the analysis is found in the "embeddings and analysis" file
The functions and classes are found in function_classes file

The model itsself is saved as a keras model, and is the onyl keras file in here
The training procedure of the model is found in the training script, and the training data is found in the TRAINING folder
