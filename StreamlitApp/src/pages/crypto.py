import pathlib
import utils.display as udisp

import streamlit as st
import core.crypto.priceApp as CalcEngine

def write():
    #udisp.title_awesome("Crypto Currency Data Analyzer")
    CalcEngine.calc_main()