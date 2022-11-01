import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from mpl_toolkits import mplot3d
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
        thead tr th:first-child {display:none}
        tbody th {display:none}
        [data-testid=column]:nth-of-type(2)
        [data-testid=stVerticalBlock]{gap: 0rem;}
        #root > div:nth-child(1) > div > div > div > div > section >
        div {padding-top: 1rem;}
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
        df = df.dropna()
        df = df.rename(columns={'mpg':'MPG'})
        return df
elem = 'car'
# varName = list(df)
# Specify order of menu
varName = ['MPG','acceleration','weight','cylinders','displacement','horsepower']
##################################################################################################
df = loadData()
##################################################################################################
# Choose the columns
def chooseColumns(x1, x2, yname):
#Store relevant columns as variables
        X = df[[x1,x2]].values.reshape(-1, 2)
        y = df[[yname]].values.reshape(-1, 1)
        return X,y
# Graph xvar vs y

def twoDscatter(xname, yname):     
        fig = plt.figure()
        plt.scatter(df[xname],df[yname],color='black')
        # plt.xlabel(xname.capitalize(),fontsize=45)
        # plt.ylabel(yname.capitalize(),fontsize=45)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20) 
        plt.xticks(rotation=45)   
        # plt.xticks(np.arange(low, high, delta/2))
        return fig
# Fit a least squares multiple linear regression model
def polynomReg(X, y):
        linModel = LinearRegression()
        linModel.fit(X,y)
        return linModel
def checkSign(val):
        unval = -val
        if val < 0:
                return " - " + str(unval)
        return " + " + str(val)
# Write the least squares model as an equation
def getFormula(x1name,x2name,yname,linModel):
        x1coef = round(linModel.coef_[0][0],3)
        x2coef = round(linModel.coef_[0][1],3)
        
        formula_text = '\widehat{\\text{' + yname + '}} = ' + str(round(linModel.intercept_[0],3))
        formula_text += checkSign(x1coef) + ' * (\\text{' + x1name + '})' +  checkSign(x2coef) + ' * (\\text{' + x2name + '})'
        return formula_text
# 3D graph
formula_text = '\( \widehat{\text{VAR}} = \)'
def get3Dgraph(x1name, x2name, yname, linModel,angle=50):
       
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        #plot the points
        ax.scatter3D(df[x1name],df[x2name],df[yname],color="Black");
        #plot the regression as a plane
        x1Delta, x2DeltaWeight = np.meshgrid(np.linspace(df[x1name].min(),df[x1name].max(),2), np.linspace(df[x2name].min(),df[x2name].max(),2))
        yDeltaMPG = (linModel.intercept_[0] + linModel.coef_[0][0]*x1Delta + linModel.coef_[0][1]*x2DeltaWeight) 
        ax.plot_surface(x1Delta, x2DeltaWeight, yDeltaMPG, alpha=0.5)
        ax.set_xlim(max(df[x1name]), min(df[x1name]))  # decreasing
        #Axes labels
        ax.set_xlabel(x1name.capitalize(), fontsize = 14)
        ax.set_ylabel(x2name.capitalize(), fontsize = 14)
        ax.set_zlabel(yname.capitalize(), fontsize = 14)
        # ax.set_xticklabels(fontsize = 10)
        # ax.set_title('MLR for '+ yname + ' by '  + x1name +' and ' + x2name)
        #Set the view angle
        ax.view_init(30,90-angle)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        font = {'size': 9}
        ax.tick_params('z', labelsize=font['size'])
        # ax.set_zticks(fontsize=9)

        # ax.set_xlim(28,9)
        return fig
def predictor(x1name, x2name, yname, linModel):
        x1median = round(df[x1name].median(),3)
        x2median = round(df[x2name].median(),3)
        yMultyPredicted = linModel.predict([[x1median,x2median]])
        sentence = 'Predicted ' + str(yname) + ' for a ' + elem + ' with \n'+ x1name + ' = ' + str(x1median) + ' and ' 
        sentence += x2name + ' = ' + str(x2median) + ' using the multiple linear regression formula is ' + str(round(yMultyPredicted[0][0],3)) + '.'
        return sentence
#### Setup streamlit gui

# X, y = chooseColumns(x1name, x2name, yname)
col1, col2 = st.columns([1,3])
with col1:
        yname = st.selectbox(
        'Dependent feature', varName 
        )    
        x1name = st.selectbox(
                'First independent feature', filter(lambda w: (w != yname), varName)
        )
        st.pyplot(twoDscatter(x1name, yname), ignore_streamlit_theme=True)
        # polynomReg(x1name,yname) or fig?
        # linModel1 = polynomReg(x1name,yname)
        # fig1 = twoDscatter(1, x1name, yname)
        x2name = st.selectbox(
                'Second independent feature', filter(lambda w: (w != yname and w != x1name), varName)
        )  
        st.pyplot(twoDscatter(x2name, yname), ignore_streamlit_theme=True)
        angle = st.slider('Horizontal rotation angle', 0, 90,40) 
with col2:
        # Display graph caption
        X, y = chooseColumns(x1name,x2name, yname)
        linModel = polynomReg(X, y)
        st.pyplot(get3Dgraph(x1name,x2name,yname,linModel,angle), ignore_streamlit_theme=True)
        st.latex(getFormula(x1name,x2name,yname,linModel))
        st.latex('\,')
        st.write(predictor(x1name, x2name, yname, linModel))
# Toggles off the Alt-text box at the bottom of the page -- default is text on
# altText = {['MPG','acceleration','weight','cylinders','displacement','horsepower']}
text_hider = st.checkbox('Hide description')
if text_hider:
        st.write("")
else:
        description = "Description: NEEDS TO BE ALGORITHMIC FROM ABOVE"
        st.write(description)
