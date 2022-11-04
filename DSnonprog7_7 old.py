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
        [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{gap: 1rem;}
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
# All other code is dependent on this dataframe definition
def loadData():
        df = pd.read_csv('customer_churn.csv', usecols=['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary', 'churn'])
        df.groupby(by='churn').size()   
        return df
df = loadData()
#####################
varName = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary', 'churn']
colLabels = ['Credit score', 'Age', 'Tenure', 'Balance', 'Number of products', 'Estimated salary', 'Customer churn']
p = sns.countplot(data=df, x='churn')
p.set_xlabel('Customer churn', fontsize=14)
p.set_ylabel('Count', fontsize=14)
# plt.show()
p = sns.boxplot(data=df, x='churn', y='credit_score')
p.set_ylabel('Credit score', fontsize=14)
p.set_xlabel('Customer churn', fontsize=14)
# plt.show()
p = sns.boxplot(data=df, x='churn', y='tenure')
p.set_ylabel('Tenure', fontsize=14)
p.set_xlabel('Customer churn', fontsize=14)
# plt.show()
p = sns.boxplot(data=df, x='churn', y='age')
p.set_ylabel('Age', fontsize=14)
p.set_xlabel('Customer churn', fontsize=14)
# plt.show()
p = sns.countplot(data=df, x='products_number', 
                  hue='churn')
p.set_xlabel('Number of products', fontsize=14)
p.set_ylabel('Count', fontsize=14)
# plt.show()
################
p = sns.scatterplot(data=df, x='age', y='churn', 
                    alpha=0.5)
p.set_xlabel('Age', fontsize=14)
p.set_ylabel('Churn (1=yes, 0=no)', fontsize=14)
# plt.show()

# Fit regression model and print coefficients
X = df[['age']].values.reshape(-1, 1)
y = df[['churn']].values.reshape(-1, 1).astype(int)

logisticModel = LogisticRegression()
logisticModel.fit(X,np.ravel(y.astype(int)))

print('Slope coefficient:', logisticModel.coef_)
print('Intercept coefficient:', logisticModel.intercept_)

# Plot fitted logistic regression model
p = sns.regplot(data=df, x='age', y='churn',
                logistic=True, ci=False, 
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'black'})
p.set_xlabel('Age', fontsize=14)
p.set_ylabel('Churn (1=yes, 0=no)', fontsize=14)
# plt.show()

################
X = df[['balance', 'age', 'credit_score']].values.reshape(-1, 3)
y = df[['churn']].values.reshape(-1, 1).astype(int)

logisticModel = LogisticRegression()
logisticModel.fit(X,np.ravel(y.astype(int)))

print('Slope coefficient:', logisticModel.coef_)
print('Intercept coefficient:', logisticModel.intercept_)

# Make predictions using logistic regression
print(logisticModel.predict_proba([[10000, 21, 650]]))

# plt.subplot(rows, columns, plot index)
plt.rcParams["figure.figsize"] = (15,6)

plt.subplot(1, 3, 1)
p = sns.scatterplot(data=df, x='balance', y='age', 
                    hue='churn', alpha=0.5)
p.set_xlabel('Balance', fontsize=14)
p.set_ylabel('Age', fontsize=14)

plt.subplot(1, 3, 2)
p = sns.scatterplot(data=df, x='balance', y='credit_score', 
                    hue='churn', alpha=0.5)
p.set_ylabel('Balance', fontsize=14)
p.set_xlabel('Credit score', fontsize=14)

plt.subplot(1, 3, 3)
p = sns.scatterplot(data=df, x='age', y='credit_score',
                  hue='churn', alpha=0.5)
p.set_xlabel('Age', fontsize=14)
p.set_ylabel('Credit score', fontsize=14)
# plt.show()
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
                'First customer feature', [] 
        )
        # # Select first indendent variable      
        # x1name = st.selectbox(
        #         'First independent feature',  varName # add lambda filter including yname if generalized
        # )
        # # Select second indendent variable     
        # x2name = st.selectbox(
        #         'Second independent feature', filter(lambda w:  w != x1name, varName)# include yname if generalized 
        # ) 
        # st.pyplot(get2Dscatter(x1name, x2name, yname, 0), ignore_streamlit_theme=True) 
        # # Graph 2C scatter plots
        # st.pyplot(get2Dscatter(x1name, x2name, yname, 1), ignore_streamlit_theme=True)
          
# with col2:
        # Display 3D graph
        # st.plotly_chart(get3Dgraph(x1name,x2name,yname, degree), config=config, ignore_streamlit_theme=True)
        # # Display regression formula
        # st.latex(getFormula(x1name, x2name, yname, degree))
        # # Display predictor sentence for median x values
        # st.write(predictor(x1name, x2name, yname, degree))
       
# Toggles off the Alt-text box at the bottom of the page -- default is to have text showing
# altText = {['MPG','acceleration','weight','cylinders','displacement','horsepower']}
# text_hider = st.checkbox('Hide description')
# if text_hider:
#         st.write("")
# else:
#         description = getAltText(x1name, x2name, yname, degree)
#         st.write(description)