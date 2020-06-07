# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:28:20 2020

@author: Alit
"""

    """
    Specify Hyperparameters
    """
    object_name = "chair"
    data_dir = "3DShapeNets/volumetric_data/" \
               "{}/30/train/*.mat".format(object_name)
    gen_learning_rate = 0.0025
    dis_learning_rate = 10e-5
    beta = 0.5
    batch_size = 1
    z_size = 200
    epochs = 10
    MODE = "train"