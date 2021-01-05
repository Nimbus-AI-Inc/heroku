import streamlit as st
from datetime import datetime

# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py
from PIL import Image

import core.compound.neural_style.style as style

import numpy as np

import utils.display as display
import utils.globalDefine as globalDefine

'''
    https://www.thecalculatorsite.com/finance/calculators/compoundinterestcalculator.php
'''

import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import core.compound.neural_style.utils
import core.compound.neural_style.transformer_net 

import core.compound.neural_style.vgg
import core.compound.neural_style.utils
import streamlit as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def yearly_compound_interest(P, R, T):   
    CI = P * (pow((1 + R / 100), T)) 
    return CI

def monthly_compound_interest(P, R, T):   
    CI = P * ((1 + R / 12) ** (12 * T))
    return CI

def calc_main(title, subtitle):
    st.sidebar.title(title)
    st.sidebar.info(
        subtitle
    )
    st.title('MosAIc: An AI Image Style Editor')

    img = st.sidebar.selectbox(
        'Select Image',
        ('amber.jpg', 'cat.png', 'MIT.jpg')
    )

    style_name = st.sidebar.selectbox(
        'Select Style',
        ('candy', 'mosaic', 'rain_princess', 'udnie')
    )


    model= "core/compound/neural_style/saved_models/" + style_name + ".pth"
    input_image = "core/compound/neural_style/images/content-images/" + img
    output_image = "core/compound/neural_style/images/output-images/" + style_name + "-" + img

    st.write('### Source image:')
    image = Image.open(input_image)
    st.image(image, width=400) # image: numpy array

    clicked = st.button('Stylize')

    if clicked:
        model = style.load_model(model)
        style.stylize(model, input_image, output_image)

        st.write('### Output image:')
        image = Image.open(output_image)
        st.image(image, width=400)

    if st.checkbox("Show help document? "):
        display.render_md("resources/compound.md")
        