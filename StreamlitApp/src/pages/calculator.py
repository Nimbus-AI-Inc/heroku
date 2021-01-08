"""
import pathlib
import utils.display as udisp

import streamlit as st
import core.calculator.CalcEngine as CalcEngine

def write():
    #udisp.title_awesome("Text Summarizer")
    CalcEngine.calc_main()
"""

import streamlit as st

import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.title('Nimbus Words')
        st.write(
            """
## Welcome!
Please check out our **AI Text Editor** on Nimbus' [Extended Page](https://nimbus-ai-inc-extension.herokuapp.com/).

## Why is Nimbus Split Into Two Sites?
Our Machine Learning Web Applications were deployed onto Heroku's servers, which limits the slugsize to 500 MB. 

As we added more applications to Nimbus, our slugsize increased, and we were forced to split our apps onto two websites, hence the creation of the **Nimbus AI Entension**.
    """
        )