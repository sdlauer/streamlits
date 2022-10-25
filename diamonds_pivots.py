import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df = pd.read_csv(
    'diamonds_casestudy.csv'
)
st.set_page_config(
    layout='centered'
)
# Hide row index column and minimize spacing between elements
hide_table_row_index_and_adjust_spacing = '''
    <style>
    thead tr th:first-child {display:none}
    tbody th {display:none}
    [data-testid=column]:nth-of-type(2)
    [data-testid=stVerticalBlock]{gap: 0rem;}
    </style>
    '''
st.markdown(hide_table_row_index_and_adjust_spacing, unsafe_allow_html=True)
#####
# Set menu and output variable options
# Many of these features are directly from the dataset,
# so should have the same capitalization as the original data
#####

# Pivot table calculation options
aggFeatures = ('mean', 'count')
# Categorical features options
catFeatures = ( 'clarity', 'color', 'cut')
# Numerical features options
numFeatures = ['carat','depth','table','price','width','length','height']
# Categorical feature categories
# Aggregation calculation format for the pivot table
sortagg = {'mean': np.mean,'count': np.size}

# Make pivot table with menu choices
def pivotTable(val='price', indx='cut', cols='color', sortby=np.mean):
    return df.pivot_table(
        values=val,
        index=indx,
        columns=cols,
        aggfunc=sortby,
        ).rename_axis(None, axis=1).reset_index()

# Page has 4 menu selectors and a pivot table
col1, col2 = st.columns([1,4])
with col1:
    numerical2 = st.selectbox(
        'Numerical pivot', numFeatures
    )
    agg = st.selectbox(
        'Aggregation', aggFeatures
    )
    genre2 = st.selectbox(
        'Categorical feature 1', catFeatures
    )
    catfeatures2 = filter(lambda x: x != genre2, catFeatures)
    genre3 = st.selectbox(
        'Categorical feature 2', catfeatures2
    )
with col2:
    # Display table heading
    st.subheader(numerical2.capitalize() + ' group ' + agg + 's for '
        + genre2 + ' and '+ genre3)
    # Display a static table
    st.table(pivotTable(val=numerical2, indx=genre2, cols=genre3, sortby=sortagg[agg]))
