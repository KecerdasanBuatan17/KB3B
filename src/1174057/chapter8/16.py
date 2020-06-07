# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:31:43 2020

@author: Alit
"""

	tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

