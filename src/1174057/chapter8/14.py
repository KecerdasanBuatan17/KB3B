# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:30:09 2020

@author: Alit
"""

    input_layer = Input(shape=(1, 1, 1, z_size))
    generated_volumes = generator(input_layer)
    validity = discriminator(generated_volumes)
    adversarial_model = Model(inputs=[input_layer], outputs=[validity])
    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)