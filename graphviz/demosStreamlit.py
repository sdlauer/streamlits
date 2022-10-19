import streamlit as st
# import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
# import graphviz as graphviz
st.graphviz_chart('''
    digraph {
        1 -> 2
        1->3
        1->4
        2->5
        3->5
        4->6

        6->5
    }
''')
col1, col2 = st.columns([2,3])
with col1:
    start_color, end_color = st.select_slider(
        'Select a range of color wavelength',
        options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
        value=('red', 'blue'))
    c12 = st.write('You selected wavelengths between', start_color, 'and', end_color)
    color1 = st.radio(
        'Select a color of the rainbow',
        ('red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'))
    c13 = st.write('My favorite color is', color1)
    c11 = st.latex(r'''
        a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
        \sum_{k=0}^{n-1} ar^k =
        a \left(\frac{1-r^{n}}{1-r}\right)
        ''')
with col2:
    age = st.slider('How old are you?', 0, 120, 25)

    c22 = st.write("I'm ", age, 'years old')
    c23 = st.markdown('Streamlit is **_really_ cool**.')
