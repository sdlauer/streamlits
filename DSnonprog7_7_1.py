import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
# st.set_page_config(layout="wide")
# hides Streamlit footer and hamburger header 
hide = '''
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        body {overflow: hidden;}
        div.block-container {padding-top:1rem;}
        div.block-container {padding-bottom:1rem;}
        thead tr th:first-child {display:none}
        tbody th {display:none}
        [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{gap: 0rem;}
        [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{gap: 0rem;}
        }
        #root > div:nth-child(1) > div > div > div > div > section >
        div {padding-top: 1rem;}
        </style>
        '''
st.markdown(hide, unsafe_allow_html=True)

# Store the data set so it doesn't have to reload -- needs a function to work
@st.cache
##################################################################################################
# Get the dataset -- this section is specific to the dataset loaded.  
# All other code is dependent on this dataframe definition and horiz/vert labels
def loadData():
        df = pd.read_csv('customer_churn.csv', usecols=['churn', 'products_number','credit_score', 'age', 'tenure', 'balance', 'estimated_salary'])
        df.groupby(by='churn').size()   
        return df
df = loadData()

colDict = {'Number of products': 'products_number', 'Credit score': 'credit_score', 'Age': 'age', 'Tenure': 'tenure', 'Balance': 'balance', 'Estimated salary': 'estimated_salary'}
countPltLabel = 'Count'
boxPltLabel = 'Customer churn'
bigVars = ['Estimated salary', 'Balance'] # ticks > 1000
def getGraphSummary(varLabel):
        var = colDict[varLabel]
        if var == 'products_number':
                summary = pd.crosstab(df['churn'], df[var])
        else:
                summary = (df.groupby('churn')[var].describe())
                summary.columns = ["Count","Mean","Std", "Min", "Q1", "Median", "Q3", "Max"]
                summary = summary[["Min", "Q1", "Median", "Q3", "Max"]].apply(lambda s: s.apply('{0:.0f}'.format))
        return summary


##################################################################################################


# Make boxplot or histogram from one column of data
def getPlot(vertLabel):
        xvar = colDict[vertLabel]
        fig = plt.figure()
        # fig.set_size_inches(3.5, 3)
        if xvar == 'products_number':
                scatterPlot = sns.countplot(data=df, x=xvar, hue='churn')
                scatterPlot.set_ylabel(vertLabel, fontsize=18)
                scatterPlot.set_xlabel(countPltLabel, fontsize=18)
        else:
                scatterPlot = sns.boxplot(data=df, x='churn', y=xvar)
                scatterPlot.set_ylabel(vertLabel, fontsize=18)
                scatterPlot.set_xlabel(boxPltLabel, fontsize=18)
        if vertLabel in bigVars:
                ylabels = ['{:,.0f}'.format(y) + 'K' for y in scatterPlot.get_yticks()/1000]
                scatterPlot.set_yticklabels(ylabels)
        return fig
###############

col1, col2 = st.columns([2,5])
with col1:
        ######
        # Customer feature
        var1 = st.selectbox(
                'First customer feature', colDict.keys() 
        )
with col2:
        # Plot boxplot or histogram for second feature   
        st.pyplot(getPlot(var1))#, ignore_streamlit_theme=True)
text_hider = st.checkbox('Hide description')
if not text_hider:
        st.dataframe(getGraphSummary(var1))
