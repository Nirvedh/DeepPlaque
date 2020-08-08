This repositiory contains code used in the paper "Deep Learning for Carotid Plaque Segmentation using a Dilated U-Net Architecture"

model.py is the main file with dilated model and you may use your own training script with the model.

This file was adapted from https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution

train.py is the training file used in the paper for semi-automatic if you wish to use it. It requires crop.py and resize.py that use simpleITK.

train_orginal.py was used for automatic setups training

hdf5 files provide trained weights in the semi-automatic setup

Please reach out to nirvedh@gmail.com if you have any questions


