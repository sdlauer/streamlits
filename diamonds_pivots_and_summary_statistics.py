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

#####
# Set menu and output variable options
# Many of these features are directly from the dataset,
# so should have the same capitalization as the original data
#####

# Pivot table calculation options
aggFeatures = ('mean', 'count')
# Categorical features options
catFeatures = ( 'clarity', 'color', 'cut')
# Order the cut category from highest to lowest quality
custom_cut = {'Fair': 4,'Good': 3,'Ideal': 0,'Premium': 1,'Very Good': 2}
# Numerical features options
numFeatures = ['carat','depth','table','price','width','length','height']
# Categorical feature categories
feature_dict ={
    'clarity': ['I1',  'IF',  'SI1',  'SI2',  'VS1',  'VS2',  'VVS1',  'VVS2'],
    'color': ['D','E','F','G','H','I','J'],
    'cut': ['Ideal','Premium','Very Good','Good','Fair'],
    }
# Aggregation calculation format for the pivot table
sortagg = {'mean': np.mean,'count': np.size}
# Hide row index column and minimize spacing between elements
hide_table_row_index_and_adjust_spacing = '''
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    body {overflow: hidden;}
    div.block-container {padding-top:1rem;}
    div.block-container {padding-bottom:1rem;}
    thead tr th:first-child {display:none}
    tbody th {display:none}
    [data-testid=column]:nth-of-type(2)
    [data-testid=stVerticalBlock]{gap: 0rem;}
    #root > div:nth-child(1) > div > div > div > div > section >
        div {padding-top: 1rem;}
    </style>
    '''
st.markdown(hide_table_row_index_and_adjust_spacing, unsafe_allow_html=True)

# Make pivot table with menu choices
def pivotTable(val='price', indx='cut', cols='color', sortby=np.mean):
    return df.pivot_table(
        values=val,
        index=indx,
        columns=cols,
        aggfunc=sortby,
        ).rename_axis(None, axis=1).reset_index()

# Make Summary statistics table with menu choices
def descriptiveStats(num='price', cat='cut'):
    return df[[cat,num]].groupby(cat).agg(
        # Get mean of the numerical column for each group
        Mean=(num, np.mean),
        # Get median of the duration column for each group
        Median=(num, np.median),
        # Get size of the duration column for each group
        Count=(num, np.size)).rename_axis(None,
            axis=1).reset_index().sort_values(by=[cat],
            key=lambda x: x.map(custom_cut)).rename(columns={cat:cat.capitalize()})

# Set page tabs for display
tab1, tab2 = st.tabs(['Summary statistics', 'Pivot table'])

# First tab has two columns:  2 menu selectors and a summary statistics table
with tab1:
    col1, col2 = st.columns([1,3])
    with col1:
        numerical1 = st.selectbox(
            'Numerical feature', numFeatures
        )
        genre1 = st.selectbox(
            'Categorical feature', catFeatures
        )
    with col2:
        # Display table caption
        st.subheader('Summary statistics for '+ numerical1 + ' by '  + genre1)
        # Display a static table
        st.table(descriptiveStats(numerical1,genre1))

# Second tab has two columns:  4 menu selectors and a pivot table
with tab2:
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
        # Display table caption
        st.subheader('Pivot table of ' + numerical2 + ' ' + agg + 's for '
            + genre2 + ' and '+ genre3)
        # Display a static table
        st.table(pivotTable(val=numerical2, indx=genre2, cols=genre3,
            sortby=sortagg[agg]))
