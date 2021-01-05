import pathlib
import utils.display as udisp

import streamlit as st
import core.compound.neural_style.CalcEngine as CalcEngine

def write():
    udisp.title_awesome("Compound Interest Calclator")
    CalcEngine.calc_main("Image Styler", "An Image Styler")

    st.write("@aditya")
