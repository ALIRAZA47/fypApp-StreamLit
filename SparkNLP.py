import pandas as pd
import streamlit as st

def func1():
    df = pd.DataFrame(['Geeks', 'For', 'Geeks', 'is', 
                'portal', 'for', 'Geeks'])
    st.dataframe(df)