import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
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
varName = ['acceleration','weight','cylinders','displacement','horsepower'] # MPG was first entry of array

##################################################################################################
# Set recurrent variables
yname = 'MPG'
x1name = varName[0]
x2name = varName[1]
df = loadData()
##################################################################################################
# Choose the columns
def setVariables(x1, x2, yvar):
#Store relevant columns as variables
        X = df[[x1,x2]].values.reshape(-1, 2)
        y = df[[yvar]].values.reshape(-1, 1)
        return (X,y)
# Store relevant columns as variables
X, y = setVariables(x1name, x2name, yname)
# Graph xvar vs y
def twoDscatter(x1name, x2name, yname, indx):  
        if indx == 0:
                xname = x1name
        else:
                xname = x2name
        X, y = setVariables(x1name, x2name, yname)     
        fig = plt.figure()
        plt.scatter(df[xname],df[yname],color='black')
        plt.xlabel(xname.capitalize(),fontsize=45)
        plt.ylabel(yname.capitalize(),fontsize=45)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20) 
        plt.xticks(rotation=45)   
        # plt.xticks(np.arange(low, high, delta/2))
        return fig
def linReg(X, y):
        linModel = LinearRegression()
        linModel.fit(X,y)
        return linModel

# Fit a least squares multiple linear regression model with degree = deg
def polynomReg(X, y, deg):
        #Fit a quadratic regression model using acceleration and weight
        polyFeatures = PolynomialFeatures(degree=deg, include_bias=False)
        xpoly = polyFeatures.fit_transform(X, y)
        polyModel = LinearRegression()
        polyModel.fit(xpoly,y)
        return polyModel
# Convert a number to a string and set just one sign -- "+ -" goes to "-"
def checkSign(val):
        if val < 0:
                return " - " + str(-val)
        return " + " + str(val)
# Write the least squares model as an equation
def getFormula(x1name, x2name, yname ,deg):
        if deg == 1:
                # Get coefficients degree 1
                linModel = linReg(X, y)
                a_int = round(linModel.intercept_[0],3)
                a0 = round(linModel.coef_[0][0],3)
                a1 = round(linModel.coef_[0][1],3) 
        #Write the polynom regression as an equation          
                formula_text = '\widehat{\\text{' + yname + '}} = ' + str(a_int) 
                formula_text += checkSign(a0) + ' * (\\text{' + x1name + '})' +  checkSign(a1) + ' * (\\text{' + x2name + '})'
        else:
                # Get coefficients degree 2
                polyModel = polynomReg(X, y, deg)
                a_int = round(polyModel.intercept_[0],3)
                a0 = round(polyModel.coef_[0][0],3)
                a1 = round(polyModel.coef_[0][1],3)  
                a2 = round(polyModel.coef_[0][2],3)
                a3 = round(polyModel.coef_[0][3],3)
                a4 = round(polyModel.coef_[0][4],3)
        # Write the polynom regression as an equation
                formula_text = '\widehat{\\text{' + yname + '}} = ' + str(a_int)
                formula_text += checkSign(a0) + ' * (\\text{' + x1name + '})' +  checkSign(a1) + ' * (\\text{' + x2name + '})'
                formula_text += checkSign(a2) + ' * (\\text{' + x1name + '\\^2})' 
                formula_text += checkSign(a3) +' * (\\text{' + x1name + '}) * (\\text{' + x2name + '})'
                formula_text += checkSign(a4) + ' * (\\text{' + x1name + '\\^2})'
        return formula_text
# 3D graph
def get3Dgraph(x1name, x2name, yname, deg ):# pm = polyModel
        X,y = setVariables(x1name, x2name, yname)
# Set dimensions of mesh and make surface polynomial mesh
        min1 = X[:,0].min()
        min2 = X[:,1].min()
        max1 = X[:,0].max()
        max2 = X[:,1].max()
        step1 = (max1 - min1)/10
        step2 = (max2 - min2)/10
        xg, yg = np.meshgrid(np.arange(min1, max1, step1), np.arange(min2, max2, step2))
        if deg == 1: 
#Fit a least squares multiple linear regression model for degree one
                lm = linReg(X, y)
                a_int  = lm.intercept_[0]
                a0 = lm.coef_[0][0]
                a1 = lm.coef_[0][1]
# Predict the linear mesh surface
                zg = a_int  + a0*xg + a1*yg
        else:
# Fit a least squares multiple linear regression model for degree 2
                pm = polynomReg(X, y, deg)
                a_int  = pm.intercept_[0]
                a0 = pm.coef_[0][0]
                a1 = pm.coef_[0][1]
                a2 = pm.coef_[0][2]
                a3 = pm.coef_[0][3]
                a4 = pm.coef_[0][4]
# Predict the quadratic mesh surface -- numpy doesn't like variable components
#       be sure to get coefficients first
                zg = a_int  + a0*xg + a1*yg + a2*xg**2 + a3*xg*yg + a4*yg**2
# Set scatterplot points
        dim = len(xg)*len(yg)
        a = np.reshape(xg, dim)
        b = np.reshape(yg, dim)
        c = np.reshape(zg, dim)
# Set up and draw the background grid
        fig = go.Figure()
# Draw a 3D scatterplot of data points
        fig.add_trace(go.Scatter3d(x=df[x1name], y=df[x2name], z=df[yname], mode='markers'))
# Draw the 3D surface mesh
        fig.add_trace(go.Mesh3d(x=a, y=b, z=c, color='black', opacity=0.5))
        # Set marker border
        fig.update_traces(
                marker=dict(size=8, line=dict(width=5, color='darkblue')),
                selector=dict(mode='markers'))
        # Set angle (tricky to work with)
        camera = dict(eye=dict(x=1.25, y=1.25, z=.5))
        # Set axis names and add the rotation info
        fig.update_layout(
                scene = dict(
                xaxis_title= x1name.capitalize(),
                yaxis_title= x2name.capitalize(),
                zaxis_title= yname.capitalize()),
                width=500,
                height=500,
                scene_camera=camera
        )
        return fig
def predictor(x1name, x2name, yname, deg):
        # Use medians of independent variables to make a prediction statement
        # X = df[[x1name,x2name]].values.reshape(-1, 2)
        # y = df[[yname]].values.reshape(-1, 1)
        x1median = round(df[x1name].median(),3)
        x2median = round(df[x2name].median(),3)
        if deg == 1:
                linModel = LinearRegression()
                linModel.fit(X,y)
                yPredicted = linModel.predict([[x1median,x2median]])
        else:
                polyModel = polynomReg(X, y, deg)
                polyFeatures = PolynomialFeatures(degree=deg, include_bias=False)
                polyInputs = polyFeatures.fit_transform([[x1median,x2median]])
                yPredicted = polyModel.predict(polyInputs)
        sentence = 'Predicted ' + str(yname) + ' for a ' + elem + ' with \n'+ x1name + ' = ' + str(x1median) + ' and ' 
        sentence += x2name + ' = ' + str(x2median) + ' using a degee ' + str(deg) + ' multiple linear regression formula is ' + str(round(yPredicted[0][0])) + '.'
        return sentence
     

########################
#### Setup streamlit gui
# Specify order of menu

col1, col2 = st.columns([1,3])
with col1:
        ###### Use if model is generalized
        # yname = st.selectbox(
        # 'Dependent feature', varName 
        # )  
        # Current dependent variable is 'MPG'
        ######
        # Select first indendent variable      
        x1name = st.selectbox(
                'First independent feature', filter(lambda w: (w != yname), varName)
        )
        # Select second indendent variable 
        st.pyplot(twoDscatter(x1name, x2name, yname, 0), ignore_streamlit_theme=True)
        x2name = st.selectbox(
                'Second independent feature', filter(lambda w: (w != yname and w != x1name), varName)
        )  
        # Select degree of model
        st.pyplot(twoDscatter(x1name, x2name, yname, 1), ignore_streamlit_theme=True)
        degree = st.selectbox(
                'Model degree', [1, 2] 
        )  
with col2:
        # Display regression formula
        st.latex(getFormula(x1name, x2name, yname, degree))
        # Display predictor sentence for median x values
        st.write(predictor(x1name, x2name, yname, degree))
        # Display 3D graph
        st.plotly_chart(get3Dgraph(x1name,x2name,yname, degree), use_container_width=True,  ignore_streamlit_theme=True)
# Toggles off the Alt-text box at the bottom of the page -- default is to have text showing
# altText = {['MPG','acceleration','weight','cylinders','displacement','horsepower']}
text_hider = st.checkbox('Hide description')
if text_hider:
        st.write("")
else:
        description = "Description: NEEDS TO BE ALGORITHMIC FROM ABOVE"
        st.write(description)
