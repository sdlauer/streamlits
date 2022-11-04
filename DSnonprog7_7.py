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
        # [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{gap: 1rem;}
        [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{gap: 1rem;}
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
        df = pd.read_csv('customer_churn.csv', usecols=['churn', 'products_number','credit_score', 'age', 'tenure', 'balance', 'estimated_salary'])
        df.groupby(by='churn').size()   
        return df
df = loadData()
#####################
varName = ['churn', 'products_number','credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
colLabels = ['Customer churn', 'Number of products', 'Credit score', 'Age', 'Tenure', 'Balance', 'Estimated salary']
colDict = {'Customer churn': varName[0], 'Number of products': varName[1], 'Credit score': varName[2], 'Age': varName[3], 'Tenure': varName[4], 'Balance': varName[5], 'Estimated salary': varName[6]}

def getPlot(xLabel):
        xname = colDict[xLabel]
        fig = plt.figure()
        fig.set_size_inches(3.5, 2)
        # p = sns.countplot(data=df, x =xname) #'churn') 
        if xname in ['churn', 'products_number']:
                if xname == 'churn':
                        p = sns.countplot(data=df, x=xname)
                else:
                        p = sns.countplot(data=df, x=xname, hue='churn')
                p.set_ylabel(xLabel, fontsize=16)
                p.set_xlabel( 'Count', fontsize=16)
        else:
                p = sns.boxplot(data=df, x='churn', y=xname) #'credit_score','tenure','age')

                p.set_ylabel(xLabel, fontsize=16)
                p.set_xlabel('Customer churn', fontsize=16)
        return fig

# plt.show()
# ################
def getLogisticGraph(xLabel):
        fig = plt.figure()
        fig.set_size_inches(3,3)
        xcolname = colDict[xLabel]
        p = sns.scatterplot(data=df, x=xcolname, y='churn', 
                        alpha=0.5)
        # p.set_xlabel('Age\nother stuff', fontsize=30)
        # p.set_ylabel('Churn (1=yes, 0=no)', fontsize=30)
        # plt.show()

        # Fit regression model and print coefficients
        X = df[[xcolname]].values.reshape(-1, 1)
        y = df[['churn']].values.reshape(-1, 1).astype(int)

        logisticModel = LogisticRegression()
        logisticModel.fit(X,np.ravel(y.astype(int)))

        # print('Slope coefficient:', logisticModel.coef_)
        # print('Intercept coefficient:', logisticModel.intercept_)

        # Plot fitted logistic regression model
        p = sns.regplot(data=df, x='age', y='churn',
                        logistic=True, ci=False, 
                        scatter_kws={'alpha': 0.5},
                        line_kws={'color': 'black'})
        p.set_xlabel(xLabel, fontsize=16)
        p.set_ylabel('Churn (1=yes, 0=no)', fontsize=16)
        plt.savefig("images7_7/logReg_" + str(xname) + ".png")
        return fig

# ################
def getLogisticInfo(x1Label, x2Label):
        x1name = colDict[x1Label]
        x2name = colDict[x2Label]
        X = df[[x1name, x2name]].values.reshape(-1, 2)
        y = df[['churn']].values.reshape(-1, 1).astype(int)

        logisticModel = LogisticRegression()
        logisticModel.fit(X,np.ravel(y.astype(int)))

        info = 'Slope coefficients:', logisticModel.coef_
        info += '\nIntercept coefficient:', logisticModel.intercept_
        x1median = round(df[x1name].median(),3)
        x2median = round(df[x2name].median(),3)
        # Make predictions using logistic regression
        arr = logisticModel.predict_proba([[x1median, x2median]])
        info += '\n\nLogistic regression customer churn prediction with ' +str(x1Label.lower()) + ' and ' + str(x2Label.lower())
        # info += 'Slope coefficients = ' + str(logisticModel.coef_)
        # info += 'at median values: \nslope = ' + str(arr[0]) + ', intercept = ' + str(arr[1])
        print(info)
        return info

def getScatter(xLabel, yLabel):     
        xname = colDict[xLabel]
        yname = colDict[yLabel]
        fig = plt.figure()
        fig.set_size_inches(5,7)
        p = sns.scatterplot(data=df, x=xname, y=yname, 
                        hue='churn', alpha=0.5)
        p.set_xlabel(xLabel, fontsize=22)
        p.set_ylabel(yLabel, fontsize=22)
        plt.xticks(fontsize=20,rotation=45)
        return fig
###############

numDataPts = df['churn'].count()





def assorted():
    hmm = 0
# Set recurrent variables

# maxs = df.max()
# mins = df.min()
# miny = mins[0]
# maxy = maxs[0]
# miny = 9
# maxy = 47


# # print(maxs, mins, numDataPts)
# textInfo = {
#         'acceleration': [8, 25, 'The points are spread out, but mostly in a cluster in the middle with a smaller cluster in the lower left'],  
#         'weight': [600, 5,200, 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right'],
#         'cylinders': [3, 8, 'The points are in vertical bands above 3, 4, 5, 6, and 8 cylinders with no points plotted between the bands'],
#         'displacement': [65, 455, 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right '
#                 'with some vertical banding stripes at higher displacement levels'],
#         'horsepower': [45,230, 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right '
#                 'with some vertical banding stripes in the middle of the range']}
# mesh = ['plane', 'quadratic surface']
# def getAltText(x1name, x2name, yname, degree):
#         return ('The scene contains 9 items:  '
#         'a selection menu for model degree, 2 selection menus for the 2 independent variables, a 2D scatterplot for each independent variable, '
#         'an interactive 3D scatter plot with a surface mesh for the model, an MPG prediction equation, a summary sentence ' 
#         'for a predicted value, and this description text.  \n'
#         'All 3 plots have {count} data points and vertical y axis of the dependent variable MPG '
#         'ranging from {miny} to {maxy}. '
#         'The first scatter plot has horizontal x axis {x1name}, ranging from {minx1} to {max1}. {shape1}. ' 
#         'The second scatter plot has horizontal x axis {x2name}, ranging from {minx2} to {max2}. {shape2}. ' 
#         'The degree {deg} three-dimensional graph has points (x,y,z) = ({x1name}, {x2name}, MPG) plotted above, on, and below the regression model {mesh}.').format(
#         count=numDataPts, miny=miny, maxy=maxy, x1name=x1name, x2name=x2name, minx1=textInfo[x1name][0], max1=textInfo[x1name][1], shape1=textInfo[x1name][2],
#         minx2=textInfo[x2name][0], max2=textInfo[x2name][1], shape2=textInfo[x2name][2], deg=degree, mesh=mesh[degree-1]
#         )

    return
# xname = colLabels[0]
# yname = colLabels[1]
col1, col2, col3 = st.columns([3,4,4])
with col1:
        ######
        # Select customer feature
        xname = st.selectbox(
                'First customer feature', colLabels 
        )
        # Plot boxplot or histogram for second feature    
        st.pyplot(getPlot(xname))#, ignore_streamlit_theme=True)
        st.pyplot(getLogisticGraph(xname))
with col2:
        # Select second feature 
        yname = st.selectbox(
                'Second customer feature', filter(lambda w:  w != xname, colLabels)
        )
        # Plot boxplot or histogram for second feature
        st.pyplot(getPlot(yname))#, ignore_streamlit_theme=True)  
        st.pyplot(getLogisticGraph(yname))
with col3:
        st.pyplot(getScatter(xname, yname))#, ignore_streamlit_theme=True)    
        st.write(getLogisticInfo(xname, yname))
# Toggles off the Alt-text box at the bottom of the page -- default is to have text showing
# altText for ['Customer churn', 'Number of products', 'Credit score', 'Age', 'Tenure', 'Balance', 'Estimated salary']
text_hider = st.checkbox('Hide description')
if text_hider:
        st.write("")
else:
        description = "TODO"# getAltText(xname, yname)
        st.write(description)