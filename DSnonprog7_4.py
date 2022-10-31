import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits import mplot3d
# In lieu of https://learn.zybooks.com/zybook/DataSciencepythonDev/chapter/7/section/4?content_resource_id=80491057
# A selection of viridis colors generated @ https://hauselin.github.io/colorpalettejs/
mycolors = ('#414487,#fde725')
# hides Streamlit footer and hamburger header 
hide = '''
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        body {overflow: hidden;}
        div.block-container {padding-top:1rem;}
        div.block-container {padding-bottom:1rem;}
        </style>
        '''

st.markdown(hide, unsafe_allow_html=True)

# Store the data set so it doesn't have to reload -- needs a function to work
@st.cache
##################################################################################################
# Get the dataset -- this section is specific to the dataset loaded.  
# All other code is dependent on this dataframe definition
def loadData():
        df = pd.read_csv('mpg.csv', usecols=['mpg','acceleration','weight','cylinders','displacement','horsepower'])
        df.columns = list(df.columns)
        df = df.dropna()
        df = df.rename(columns={'mpg':'MPG'})
        return df
elem = 'car'
##################################################################################################
df = loadData()

varName = list(df.columns)
print(varName)
##################################################################################################
# Choose the columns
def chooseColumns(x1, x2, yvar):
#Store relevant columns as variables
        X = df[[x1,x2]].values.reshape(-1, 2)
        y = df[[yvar]].values.reshape(-1, 1)
        return X,y
# Graph xvar vs y
def twoDscatter(xfeatnum=1, xname=varName[1], yname=varName[0]):
        fig = plt.figure()
        plt.scatter(X[:,xfeatnum],y,color='black')
        plt.xlabel(xname.capitalize(),fontsize=14)
        plt.ylabel(yname.capitalize(),fontsize=14)
        return fig
# Fit a least squares multiple linear regression model
def polynomReg(X, y):
        linModel = LinearRegression()
        linModel.fit(X,y)
        return linModel
# Write the least squares model as an equation
def getFormula(x1var,x2var,yvar,linModel):
        formula_text = '\( \widehat{\text{' + yvar + '}} = \)' + str(round(linModel.intercept_[0],2)) + ' + ' 
        formula_text += str(round(linModel.coef_[0][0],4)) + ' * ('+ x1var + ')' + ' + ' +  str(round(linModel.coef_[0][1],2)) + ' * (' + x2var+ ')'
        return formula_text
# 3D graph
def get3Dgraph(x1var, x2var, yvar, X, y):
        linModel = polynomReg(X, y)
        formula_text = getFormula(x1var,x2var,yvar,linModel)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        #plot the points
        ax.scatter3D(X[:,0],X[:,1],y,color="Black");
        #plot the regression as a plane
        x1Delta, x2DeltaWeight = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),2), np.linspace(X[:,1].min(),X[:,1].max(),2))
        yDeltaMPG = (linModel.intercept_[0] + linModel.coef_[0][0]*x1Delta + linModel.coef_[0][1]*x2DeltaWeight) 
        ax.plot_surface(x1Delta, x2DeltaWeight, yDeltaMPG, alpha=0.5);
        #Axes labels
        ax.set_xlabel(x1var.capitalize())
        ax.set_ylabel(x2var.capitalize())
        ax.set_zlabel(yvar.capitalize())
        #Set the view angle
        ax.view_init(30,50)
        # ax.set_xlim(28,9)
        return linModel, fig, formula_text
def predictor(x1var, x2var, yvar, linModel):
        x1median = round(df[x1var].median(),2)
        x2median = round(df[x2var].median(),2)
        yMultyPredicted = linModel.predict([[x1median,x2median]])
        sentence = 'Predicted ' + str(yvar) + ' for a ' + elem + ' with '+ x1var + ' = ' + str(x1median) + ' and ' 
        sentence += x2var + ' = ' + str(x2median) + 'using the multiple linear regression is ' + str(round(yMultyPredicted[0][0],2)) + '.'
        return sentence
###
#  Initial setup for X, y, linear graphs, MLP equation, and MLP graph
###
name=varName[0]
xname=varName[1]
xname=varName[2]
X, y = chooseColumns(x1=varName[1],x2=varName[2],yvar=varName[0])
fig1 = twoDscatter(xfeatnum=0, xname=varName[1], yname=varName[0])
fig2 = twoDscatter(xfeatnum=1, xname=varName[2], yname=varName[0])
linModel, fig3, MLP = get3Dgraph(varName[1],varName[2],varName[0],X,y)
print(MLP)

# tab1, tab2 = st.tabs(['Polynomial regression', 'Pivot table'])

# First tab has two columns:  2 menu selectors and a summary statistics table
# with tab1:
yname = st.selectbox(
        'Dependent feature', varName 
)    
col1, col2 = st.columns([1,3])
with col1:
        x1name = st.selectbox(
                'First independent feature', filter(lambda x: (x != yname), varName)
        )
        # polynomReg(x1name,yname) or fig?
        # linModel1 = polynomReg(x1name,yname)
        fig1 = twoDscatter(1, x1name, yname)
        st.pyplot(fig1, ignore_streamlit_theme=True)
        x2name = st.selectbox(
                'Second independent feature', filter(lambda x: (x != yname and x != x1name), varName)
        )
        fig2 = twoDscatter(1, x2name, yname)
        st.pyplot(fig2, ignore_streamlit_theme=True)
        # X,y = chooseColumns(yname, x1name, x2name)
with col2:
        # Display graph caption
        st.subheader('Multiple linear regression for '+ yname + ' by '  + x1name +' and ' + x2name)
        # fig3 = get3Dgraph(x1name,x2name,yname,X,y,linModel)
        # st.pyplot(fig3, ignore_streamlit_theme=True)
        # st.latex('r' + getFormula(x1name,x2name,yname))

# Second tab has two columns:  4 menu selectors and a pivot table
# with tab2:
#     col1, col2 = st.columns([1,4])
#     with col1:
#         nxcarF = st.selectbox(
#             'Numerical pivot', numFeatures
#         )
#         agg = st.selectbox(
#             'Aggregation', aggFeatures
#         )
#         genre2 = st.selectbox(
#             'Categorical feature 1', catFeatures
#         )
#         catfeatures2 = filter(lambda x: x != genre2, catFeatures)
#         genre3 = st.selectbox(
#             'Categorical feature 2', catfeatures2
#         )
#     with col2:
#         # Display table caption
#         st.subheader('Pivot table of ' + numerical2 + ' ' + agg + 's for '
#             + genre2 + ' and '+ genre3)

