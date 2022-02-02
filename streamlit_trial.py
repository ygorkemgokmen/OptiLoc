#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl

st.write("This is a kepler.gl map in streamlit")

map_1 = KeplerGl(height=400)
keplergl_static(map_1)


