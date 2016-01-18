#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import brainstorm as bs
import numpy as np
from PIL import Image

network = bs.Network.from_hdf5('mnist_pi_best.hdf5')
image = Image.open("test_4.jpg")
data = np.array(image).reshape(
        image.size[0], image.size[1], 3).dot(
        [0.2, 0.7, 0.1]).reshape(
        image.size[0], image.size[1], 1) / 255
network.provide_external_data(
        {'default': np.array([[data]])},
        all_inputs=False)
network.forward_pass(training_pass=False)
classification = network.get('Output.outputs.predictions')[0][0]
print(np.argmax(classification))
print(classification)
