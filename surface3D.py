
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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
x1name = 'acceleration'
x2name = 'weight'
yname = 'MPG'
X = df[[x1name,x2name]].values.reshape(-1, 2)
y = df[[yname]].values.reshape(-1, 1)
# Set dimensions of mesh and make surface polynomial mesh
min1 = X[:,0].min()
min2 = X[:,1].min()
max1 = X[:,0].max()
max2 = X[:,1].max()
step1 = (max1 - min1)/10
step2 = (max2 - min2)/10
xg, yg = np.meshgrid(np.arange(min1, max1, step1), np.arange(min2, max2, step2))
deg = 1
# 
if deg == 1: 
#Fit a least squares multiple linear regression model for degree one
        lm = LinearRegression()
        lm.fit(X,y)
        aint = lm.intercept_[0]
        a0 = lm.coef_[0][0]
        a1 = lm.coef_[0][1]
# Predict the linear mesh surface
        zg = aint + a0*xg + a1*yg
else:
# Fit a least squares multiple linear regression model for degree 2
        polyFeatures = PolynomialFeatures(degree=2, include_bias=False)
        xPoly = polyFeatures.fit_transform(X)
        pm = LinearRegression() 
        pm.fit(xPoly,y)
        aint = pm.intercept_[0]
        a0 = pm.coef_[0][0]
        a1 = pm.coef_[0][1]
        a2 = pm.coef_[0][2]
        a3 = pm.coef_[0][3]
        a4 = pm.coef_[0][4]
# Predict the quadratic mesh surface
        zg = aint + a0*xg + a1*yg + a2*xg**2 + a3*xg*yg + a4*yg**2
# Set scatterplot points
dim = len(xg)*len(yg)
a = np.reshape(xg, dim)
b = np.reshape(yg, dim)
c = np.reshape(zg, dim)
# Set up and draw the background grid
fig = go.Figure()
# Draws a 3D scatterplot of data points
fig.add_trace(
    go.Scatter3d(x=df[x1name], y=df[x2name], z=df[yname], mode='markers'))
# Draw the 3D surface mesh
fig.add_trace(go.Mesh3d(x=a, y=b, z=c, color='black', opacity=0.5))
# Set marker border
fig.update_traces(marker=dict(size=8, line=dict(width=5, color='darkblue')),
                selector=dict(mode='markers'))
# Set angle (tricky to work with)
name = 'eye = (x:2, y:2, z:.5)'
camera = dict(
    eye=dict(x=2, y=2, z=.5)
)
# Set axis names and add the rotation info
fig.update_layout(scene = dict(
                    xaxis_title= x1name.capitalize(),
                    yaxis_title= x2name.capitalize(),
                    zaxis_title= yname.capitalize()),
                scene_camera=camera
)
fig.show()