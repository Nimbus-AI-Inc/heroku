import streamlit as st

st.set_page_config(page_title= 'Nimbus AI Inc.',
    page_icon= '⛈')

import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('b.jpg')

import utils.display as udisp

import src.pages.home
import src.pages.about
import src.pages.calculator
import src.pages.compound
# import src.pages.scikit_image
import src.pages.stocks

MENU = {
    "Home" : src.pages.home,
    "Simple Calculator" : src.pages.calculator,
    "Calculate Compound Interest" : src.pages.compound,
    # "Scikit Image Demo" : src.pages.scikit_image,
    "Stocks" : src.pages.stocks,
    "Credits" : src.pages.about
}


def main():
    st.sidebar.title("Navigate yourself...")
    menu_selection = st.sidebar.radio("Choice your option...", list(MENU.keys()))

    menu = MENU[menu_selection]

    with st.spinner(f"Loading {menu_selection} ..."):
        udisp.render_page(menu)

    st.sidebar.info(
        "https://github.com/Avkash/demoapps"
    )
    st.sidebar.info(
        "demoapps/StreamlitApp"
    )

if __name__ == "__main__":
    main()
