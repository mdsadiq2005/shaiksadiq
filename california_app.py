import numpy as np
import joblib
import streamlit as st


st.title("california App")
obj=joblib.load("california.joblib")
model=obj['lr']
cols=obj["columns"]

st.title("california app")
In=[]
for i in cols:
    v=st.number_input(f"enter {i} value")
    In.append(v)
if st.button("click"):
    out=model.predict([In])
    st.success({out[0]})    




