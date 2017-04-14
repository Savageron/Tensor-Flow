# Tensor-Flow
Automated Intepretation of Seafloor 3D Visual Mapping Data Obtained using Underwater Robots
Third Year Individual Project

"Magnus" folder\
Contains the simplified cifar10 version of Convolutional Neural Network for processing the Iheya_n files, modified from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/06_CIFAR-10.ipynb \
-cache.py: \
-cifar10.py: \
-cifar10HvassLabs1.py: \
-dataset.py: 

"cifar10_on_Iheya" folder\
Contains the actual cifar10 version of Convolutional Neural Network for processing the Iheya_n files, but is discontinued as it is too complicated and time consuming to alter the algorithm to produce desired outputs.\
-_init_.py:\
-cifar10.py: \
-cifar10_eval.py: \
-cifar10_input.py: \
-cifar10_train.py: 

"non_tf_imgprocess" folder\
Contains the codes to process the raw images/mosaic to produce an output in the desired format for feeding into the Convolutional Neural Network.\
-databatchmixing.py: Randomly mix the order of training dataset to check if this affects the CNN results (original arrangement of dataset is species 0 in data_batch_1.txt, species 1 in data_batch_2.txt, and species 2 in data_batch_3.txt).\
-mosaic_crop.py: Crop imgaes from mosaic with coordinates defined in a csvfile. Sort them into training/test set. Combine the two bathymodiolus species as one data set. Generate null dataset. Convert text file to bin file. \
-ratiochange.py: Generate different train to test ratio dataset\
-showanimage.py: This code was created to output a cropped image of the mosaic at the coordinates of interest to check whether the image from null set that showed a bathymodiolus is indeed unlabelled.\
-sizecutting3to1ratio.py: Change the number of training data set from the 3 to 1 train to test ratio data set, but keeping the test set the same.

Author: Jin Lim
