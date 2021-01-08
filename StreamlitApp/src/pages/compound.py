import pathlib
import utils.display as udisp

import streamlit as st
import core.compound.neural_style.CalcEngine as CalcEngine

def write():
    #udisp.title_awesome("Image Styler")
    CalcEngine.calc_main()
