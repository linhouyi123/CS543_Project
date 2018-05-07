# CS543_Project
This is a course project for CS543. Code adapted from https://github.com/mvirgo/MLND-Capstone

## Dataset
Original dataset from: https://xingangpan.github.io/projects/CULane.html
Preprocessed Dataset link: [link](https://drive.google.com/open?id=1uiaKUjLtM6LrEqI9kwRrTwF-suW_u7Qj "Preprocessed Dataset")

## File Functionality
* fully_conv_NN.py defines the CNN structure.
* draw_detected_lanes.py draws predicted lanes on an test image.
* preprocess.py takes train_182.txt as input and saves the preprocessed images into pickle file format.
* train_182 specifies the directory at which the training dataset and labels locate.
* CULane_CNN_model.h5 contains the trained model and are ready to use.