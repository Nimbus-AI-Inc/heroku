"""Home page shown when the user enters the application"""
import streamlit as st

#import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.title('Nimbus AI Home Page')
        st.write(
            """
## Welcome!

Nimbus AI Inc. is a startup focused on developing free **Machine Learning Web Applications** for public use. 

Our current Apps Include:

- An AI **Image Styler** which was created using a CNN Neural Network.
- An AI **Stock Predictor** which was created using an LSTM Neural Network.
- A **Crypto Currency Data Analyzer** which provides analysis for 100 crypto currencies.
- An AI **Face Generator** which was created using NVidia's Open Source APIs, MIT's free-to-use creative liscence and a Generative Adversarial Network.
- An AI **Text Editor** which provides users summarization, tokenization, translation, synonym functionality and more.

👈 **Check them out using our side bar to the left!**

## Quick Info

As of 2020 we have recieved official Non-Profit status under the state of Deleware, and are dedicated to the cause of keeping all our services completely free for consumer use. 

If you wish to contact us please email adityashukla2015@gmail.com or check out [our GitHub](https://github.com/Nimbus-AI-Inc) if you have any specific questions related to the Machine Learning algorithms used to develop Nimbus' Library.

If you wish to utilize any of the code that has been used in our projects feel free to do so. We permit the implementation of our code for both commercial and creative intents. 

    """
        )
