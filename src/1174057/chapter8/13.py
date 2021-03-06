# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:29:18 2020

@author: Alit
"""

    """
    Create models
    """
    gen_optimizer = Adam(lr=gen_learning_rate, beta_1=beta)
    dis_optimizer = Adam(lr=dis_learning_rate, beta_1=beta)

    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)