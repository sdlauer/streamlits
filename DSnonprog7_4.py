import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
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
        [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{gap: 1rem;}
        [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{gap: 0rem; }
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
# All other code is dependent on this dataframe definition
def loadData():
        df = pd.read_csv('mpg.csv', usecols=['mpg','acceleration','weight','cylinders','displacement','horsepower'])
        df = df.dropna()
        df = df.rename(columns={'mpg':'MPG'})
        return df
elem = 'car'
varName = ['acceleration','weight','cylinders','displacement','horsepower'] # MPG was first entry of array

# Set recurrent variables
yname = 'MPG'
df = loadData()
maxs = df.max()
mins = df.min()
miny = mins[0]
maxy = maxs[0]
numDataPts = df['MPG'].count()
print(maxs, mins, numDataPts)
textInfo = {
        'acceleration': [df[varName[0]].min(), df[varName[0]].max(), 'The points are spread out, but mostly in a cluster in the middle with a smaller cluster in the lower left'],  
        'weight': [df[varName[1]].min(), df[varName[1]].max(), 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right'],
        'cylinders': [df[varName[2]].min(), df[varName[2]].max(), 'The points are in vertical bands above 3, 4, 5, 6, and 8 cylinders with no points plotted between the bands'],
        'displacement': [df[varName[3]].min(), df[varName[3]].max(), 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right '
                'with some vertical banding stripes at higher displacement levels'],
        'horsepower': [df[varName[4]].min(), df[varName[4]].max(), 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right '
                'with some vertical banding stripes in the middle of the range']}
mesh = ['plane', 'quadratic surface']
def getAltText(x1name, x2name, yname, degree):
        return ('The scene contains 9 items:  '
        'a selection menu for model degree and 2 selection menus for the 2 independent variables, a 2D scatterplot for each independent variable, '
        'a reactive 3D scatter plot with a surface mesh for the model, an MPG prediction equation, a summary sentence ' 
        'for a predicted value, and this description text.  \n'
        'All 3 plots have {count} data points and vertical y axis of the dependent variable MPG '
        'ranging from {miny} to {maxy}. '
        'The first scatter plot has horizontal x axis {x1name}, ranging from {minx1} to {max1}. {shape1}. ' 
        'The second scatter plot has horizontal x axis {x2name}, ranging from {minx2} to {max2}. {shape2}. ' 
        'The degree {deg} three-dimensional graph has points (x,y,z) = ({x1name}, {x2name}, MPG) plotted above, on, and below the regression model {mesh}.').format(
        count=numDataPts, miny=miny, maxy=maxy, x1name=x1name, x2name=x2name, minx1=textInfo[x1name][0], max1=textInfo[x1name][1], shape1=textInfo[x1name][2],
        minx2=textInfo[x2name][0], max2=textInfo[x2name][1], shape2=textInfo[x2name][2], deg=degree, mesh=mesh[degree-1]
        )
##################################################################################################
# Choose the columns
def setVariables(x1, x2, yvar):
#Store relevant columns as variables
        X = df[[x1,x2]].values.reshape(-1, 2)
        y = df[[yvar]].values.reshape(-1, 1)
        return (X,y)
# Store relevant columns as variables
# X, y = setVariables(x1name, x2name, yname)
# Graph xvar vs y
def get2Dscatter(x1name, x2name, yname, indx): 
        xname = [x1name, x2name] 
        X, y = setVariables(x1name, x2name, yname)
        fig = plt.figure()
        plt.scatter(X[:,indx],y, c='k')
        plt.xlabel(xname[indx].capitalize(),fontsize=35)
        plt.ylabel(yname.capitalize(),fontsize=35)
        plt.xticks(fontsize=20,rotation=45)
        plt.yticks(fontsize=20)   
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
                sgn = " - \\,"
                val = -val
        else:
                sgn = " + \\, "
        if abs(val) < 0.001:
                num = "{:.3e}".format(abs(val))
                num = str(num.replace('-0','-').replace('e0','e'))
                num = num[:5] + '\\times 10^{' + num[6:] + '}'
        else:
                num = round(val,3)      
        return sgn + str(num)
# Write the least squares model as an equation
def getFormula(x1name, x2name, yname ,deg):
        X, y = setVariables(x1name, x2name, yname)
        if deg == 1:
                # Get coefficients degree 1
                linModel = linReg(X, y)
                a_int = round(linModel.intercept_[0],3)
                a0 = linModel.coef_[0][0]
                a1 = linModel.coef_[0][1] 
        #Write the polynom regression as an equation          
                formula_text = '\\\\\,\\\\\,\\\\\,\\widehat{\\text{' + yname + '}} = ' + str(a_int) 
                formula_text += checkSign(a0) + '(\\text{' + x1name + '})' +  checkSign(a1) + '(\\text{' + x2name + '})\\\\\,\\\\\,'
        else:
                # Get coefficients degree 2
                polyModel = polynomReg(X, y, deg)
                a_int = round(polyModel.intercept_[0],3)
                a0 = polyModel.coef_[0][0]
                a1 = polyModel.coef_[0][1]  
                a2 = polyModel.coef_[0][2]
                a3 = polyModel.coef_[0][3]
                a4 = polyModel.coef_[0][4]
        # Write the polynom regression as an equation
                formula_text = '\\begin{align*}'
                formula_text += '\\,\\widehat{\\text{' + yname + '}} = & ' + str(a_int)
                formula_text += checkSign(a0) + '(\\text{' + x1name + '})' +  checkSign(a1) + '(\\text{' + x2name + '})\\\\'
                formula_text += ' & ' + checkSign(a2) + '(\\text{' + x1name + '})^2\\\\' 
                formula_text += ' & ' + checkSign(a3) +'(\\text{' + x1name + '})(\\text{' + x2name + '})\\\\'
                formula_text += ' & ' + checkSign(a4) + '(\\text{' + x2name + '})^2'
                formula_text += '\\end{align*}'
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
                marker=dict(size=5, line=dict(width=10, color='darkblue')),
                selector=dict(mode='markers'))
        # Set angle (tricky to work with)
        camera = dict(eye=dict(x=1.5, y=1.5, z=.1))
        # Set axis names and add the rotation info
        fig.update_layout(
                scene = dict(
                xaxis_title= x1name.capitalize(),
                yaxis_title= x2name.capitalize(),
                zaxis_title= yname.capitalize()),
                # width=600,
                height=400,
                scene_camera=camera,
                margin=dict(l=0, r=100, b=0, t=0)
        )
        return fig
def predictor(x1name, x2name, yname, deg):
        # Use medians of independent variables to make a prediction statement
        X,y = setVariables(x1name, x2name, yname)
        # X = df[[x1name,x2name]].values.reshape(-1, 2)
        # y = df[[yname]].values.reshape(-1, 1)
        x1median = round(df[x1name].median(),3)
        x2median = round(df[x2name].median(),3)
        if deg == 1:
                linModel = linReg(X, y)
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
config = {'displaylogo': False, 'displayModeBar': False}
col1, col2 = st.columns([1,3])
with col1:
        ###### Use if model is generalized to using all variables for y
        # yname = st.selectbox(
        # 'Dependent feature', varName 
        # )  
        # Current dependent variable is 'MPG'
        ######
        # Select degree of model
        degree = st.selectbox(
                'Model degree', [1, 2] 
        )
        # Select first indendent variable      
        x1name = st.selectbox(
                'First independent feature',  varName # add lambda filter including yname if generalized
        )
        # Select second indendent variable     
        x2name = st.selectbox(
                'Second independent feature', filter(lambda w:  w != x1name, varName)# include yname if generalized 
        ) 
        st.pyplot(get2Dscatter(x1name, x2name, yname, 0), ignore_streamlit_theme=True) 
        # Graph 2C scatter plots
        st.pyplot(get2Dscatter(x1name, x2name, yname, 1), ignore_streamlit_theme=True)
          
with col2:
        # Display 3D graph
        st.plotly_chart(get3Dgraph(x1name,x2name,yname, degree), config=config, ignore_streamlit_theme=True)
        # Display regression formula
        st.latex(getFormula(x1name, x2name, yname, degree))
        # Display predictor sentence for median x values
        st.write(predictor(x1name, x2name, yname, degree))
       
# Toggles off the Alt-text box at the bottom of the page -- default is to have text showing
# altText = {['MPG','acceleration','weight','cylinders','displacement','horsepower']}
text_hider = st.checkbox('Hide description')
if text_hider:
        st.write("")
else:
        description = getAltText(x1name, x2name, yname, degree)
        st.write(description)
