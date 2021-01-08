import pathlib
import utils.display as udisp

import streamlit as st
import core.stocks.final_application.app2 as CalcEngine

def write():
    #udisp.title_awesome("Stock Predictor")
    CalcEngine.calc_main()