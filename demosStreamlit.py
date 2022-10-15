import streamlit as st
# import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import graphviz as graphviz
# import matplotlib.pyplot as plt
df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))

st.dataframe(df)  # Same as st.write(df)


df = pd.DataFrame(
   np.random.randn(10, 5),
   columns=('col %d' % i for i in range(5)))
st.table(df)

st.metric(label="Temperature", value="70 °F", delta="1.2 °F")


st.graphviz_chart('''
    digraph {
        1 -> 2
        1->3
        1->4
        2->5
        3->5
        4->5
    }
''')
st.title('This is a title')
st.header('This is a header')
st.subheader('This is a subheader')
st.caption('This is a string that explains something above.')
code = '''def hello():
    print("Hello, Streamlit!")'''
st.code(code, language='python')
st.text('This is some text.')
st.latex(r'''
    a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
    \sum_{k=0}^{n-1} ar^k =
    a \left(\frac{1-r^{n}}{1-r}\right)
    ''')

# plt.viridis()
start_color, end_color = st.select_slider(
    'Select a range of color wavelength',
    options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
    value=('red', 'blue'))
st.write('You selected wavelengths between', start_color, 'and', end_color)
color1 = st.select_slider(
    'Select a color of the rainbow',
    options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
st.write('My favorite color is', color1)

age = st.slider('How old are you?', 0, 120, 25)

st.write("I'm ", age, 'years old')
st.markdown('Streamlit is **_really_ cool**.')


# rgb(253, 231, 37)
#
# #fde725
# rgb(208, 225, 28)
#
# #d0e11c
# rgb(160, 218, 57)
#
# #a0da39
# rgb(115, 208, 86)
#
# #73d056
# rgb(74, 193, 109)
#
# #4ac16d
# rgb(45, 178, 125)
#
# #2db27d
# rgb(31, 161, 135)
#
# #1fa187
# rgb(33, 145, 140)
#
# #21918c
# rgb(39, 127, 142)
#
# #277f8e
# rgb(46, 110, 142)
#
# #2e6e8e
# rgb(54, 92, 141)
#
# #365c8d
# rgb(63, 71, 136)
#
# #3f4788
# rgb(70, 50, 126)
#
# #46327e
# rgb(72, 27, 109)
#
# #481b6d
# rgb(68, 1, 84)
#
# #440154
