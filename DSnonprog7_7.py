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
imageFolder = 'images7_7/logReg_' # folder and model-specific file name prefix
numDataPts = df['churn'].count()
def getGraphSummary(varLabel):
        var = colDict[varLabel]
        if var == 'products_number':
                summary = pd.crosstab(df['churn'], df[var])
        else:
                summary = (df.groupby('churn')[var].describe())
                summary.columns = ["Count","Mean","Std", "Min", "Q1", "Median", "Q3", "Max"]
                summary = summary[["Min", "Q1", "Median", "Q3", "Max"]].apply(lambda s: s.apply('{0:.0f}'.format))
        return summary
# # print(maxs, mins, numDataPts)
# altTxtDict = {
#         'Number of products': [, 'The points are spread out, but mostly in a cluster in the middle with a smaller cluster in the lower left'],  
#         'Credit score': [, 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right'],
#         'Age': [, 'The points are in vertical bands above 3, 4, 5, 6, and 8 cylinders with no points plotted between the bands'],
#         'Tenure': [, 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right '
#                 'with some vertical banding stripes at higher displacement levels'],
#         'Balance': [, 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right '
#                 'with some vertical banding stripes in the middle of the range']
#         'Estimated salary': [, 'The points are in a concave up, crescent-shaped area from the upper left decreasing to the lower right '
#                 'with some vertical banding stripes in the middle of the range']}
# mesh = ['plane', 'quadratic surface']
def getAltText(x1name, x2name):
        return ('#### This summary will be completed when Rev1 for Tab1 is finished. \n####\n'
        'The {x1name} logistic plot s-curve is asymptotic with churn = 0 from min0 to max0 and curves up to be asymptotic with churn = 1 from min1 to max1.\n\n'
        'The scatter plot of x = {x1name} and y = {x2name} has x1name points concentrated ... and x2name points concentated ...')
        return ('The scene contains 9 items:  '
        'a selection menu for model degree, 2 selection menus for the 2 independent variables, a 2D scatterplot for each independent variable, '
        'an interactive 3D scatter plot with a surface mesh for the model, an MPG prediction equation, a summary sentence ' 
        'for a predicted value, and this description text.  \n'
        'All 3 plots have {count} data points and vertical y axis of the dependent variable MPG '
        'ranging from {miny} to {maxy}. '
        'The first scatter plot has horizontal x axis {x1name}, ranging from {minx1} to {max1}. {shape1}. ' 
        'The second scatter plot has horizontal x axis {x2name}, ranging from {minx2} to {max2}. {shape2}. ' 
        'The degree {deg} three-dimensional graph has points (x,y,z) = ({x1name}, {x2name}, MPG) plotted above, on, and below the regression model {mesh}.').format(
        count=numDataPts, miny=miny, maxy=maxy, x1name=x1name, x2name=x2name, minx1=altTxtDict[x1name][0], max1=altTxtDict[x1name][1], shape1=altTxtDict[x1name][2],
        minx2=altTxtDict[x2name][0], max2=altTxtDict[x2name][1], shape2=altTxtDict[x2name][2], deg=degree, mesh=mesh[degree-1]
        )
        return
##################################################################################################
# Format signed numbers in equations
def checkSign(val, txt): # txt = 'text' is for alt-text
        if val < 0:
                sgn = '-'
                val = -val
        else:
                sgn = ''
        if abs(val) < 0.0001:
                num = "{:.4e}".format(abs(val))
                num = str(num.replace('-0','-').replace('e0','e'))
                if txt == 'text':
                        num = num[:6] + ' times 10 to the power of ' + num[6:] 
                num = num[:6] + '\\times 10^{' + num[7:] + '}'
        else:
                num = round(val,4)   
        return sgn + str(num)

# Make boxplot or histogram from one column of data
def getPlot(vertLabel):
        xvar = colDict[vertLabel]
        fig = plt.figure()
        fig.set_size_inches(3.5, 3)
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
# Call this function to get all graphs needed for the final streamlit
# Need imageFolder made in project folder
def getLogisticGraph(horizLabel): # str attribute
        fig = plt.figure()
        fig.set_size_inches(3,2)
        ax=fig.add_axes([0,0,1,1])
        xcolname = colDict[horizLabel]
        scatterPlot = sns.scatterplot(data=df, x=xcolname, y='churn', alpha=0.5)
        scatterPlot.set_xlabel('Age\nother stuff', fontsize=18)
        scatterPlot.set_ylabel('Churn (1=yes, 0=no)', fontsize=18)
        # Fit regression model
        X = df[[xcolname]].values.reshape(-1, 1)
        y = df[['churn']].values.reshape(-1, 1).astype(int)
        logisticModel = LogisticRegression()
        logisticModel.fit(X,np.ravel(y.astype(int)))
        # Plot fitted logistic regression model
        scatterPlot = sns.regplot(data=df, x='age', y='churn',
                        logistic=True, ci=False, 
                        scatter_kws={'alpha': 0.5},
                        line_kws={'color': 'black'})
        scatterPlot.set_xlabel(horizLabel, fontsize=18)
        scatterPlot.set_ylabel('Churn (1=yes, 0=no)', fontsize=18)
        ax.set_yticks([0,0.5,1])
        if horizLabel in bigVars:
                print(horizLabel)
                xlabels = ['{:,.0f}'.format(x) + 'K' for x in scatterPlot.get_xticks()/1000]
                ax.set_xticklabels(xlabels)
        plt.xticks(fontsize=18,rotation=45)
        # Saves image in in project (variable previously defined imageFolder)
        # plt.savefig(imageFolder + str(horizLabel) + ".png", bbox_inches='tight')
        # Returns live to s l o w l y render image
        return fig


def get2VarScatter(xLabel, yLabel): # str, str attributes
        xvar = colDict[xLabel]
        yvar = colDict[yLabel]
        fig = plt.figure()
        fig.set_size_inches(3.5,6.5)
        scatterPlot = sns.scatterplot(data=df, x=xvar, y=yvar, 
                        hue='churn', alpha=0.5)
        scatterPlot.set_xlabel(xLabel, fontsize=14)
        scatterPlot.set_ylabel(yLabel, fontsize=14)
        if xLabel in bigVars:
                xlabels = ['{:,.0f}'.format(x) + 'K' for x in scatterPlot.get_xticks()/1000]
                scatterPlot.set_xticklabels(xlabels)
        if yLabel in bigVars:
                ylabels = ['{:,.0f}'.format(y) + 'K' for y in scatterPlot.get_yticks()/1000]
                scatterPlot.set_yticklabels(ylabels)
        plt.xticks(fontsize=14)#, rotation=45)
        plt.yticks(fontsize=14)#,rotation=45)
        return fig
def getLogisticInfo(x1Label, x2Label): # str, str attributes
        x1name = colDict[x1Label]
        x2name = colDict[x2Label]
        X = df[[x1name, x2name]].values.reshape(-1, 2)
        y = df[['churn']].values.reshape(-1, 1).astype(int)
        logisticModel = LogisticRegression()
        logisticModel.fit(X,np.ravel(y.astype(int)))
        aint = logisticModel.intercept_[0]
        slope0 = logisticModel.coef_[0][0]
        slope1 = logisticModel.coef_[0][1]
        info = '\\text{ Intercept: }'+ checkSign(aint,'')
        info += '\\qquad \\text{ Slopes: (} '+ checkSign(slope0, '') + ', ' + checkSign(slope1, '') + ') \\qquad \\qquad \\qquad \\qquad \\qquad\\quad\\\\'
        x1median = round(df[x1name].median(),3)
        x2median = round(df[x2name].median(),3)
        # Make predictions using logistic regression
        arr = logisticModel.predict_proba([[x1median, x2median]])
        prob0 = arr[0][0]
        prob1 = arr[0][1] 
        info += '\\,\\text{Logistic regression customer churn prediction with median values}: '
        info += '\\widehat{p} = (' + checkSign(prob0, '')+ ', '+ checkSign(prob1, '')+ ')'
        # print(info)
        return info


###############

tab1, tab2 = st.tabs(['Plots', 'Description'])
with tab1:
        col1, col2, col3 = st.columns([3,3,4])
        with col1:
                ######
                # Select first customer feature
                var1 = st.selectbox(
                        'First customer feature', colDict.keys() 
                )
                # Plot boxplot or histogram for second feature    
                st.pyplot(getPlot(var1))#, ignore_streamlit_theme=True)
                # st.pyplot(getLogisticGraph(var1)) # use to call and make images b/c logistic loads slowly
                st.image('images7_7/logReg_'+var1+'.png') # use images made with previous line
        with col2:
                # Select second customer feature 
                var2 = st.selectbox(
                        'Second customer feature', filter(lambda w:  w != var1, colDict.keys())
                )
                st.pyplot(getPlot(var2))#, ignore_streamlit_theme=True)  
                # st.pyplot(getLogisticGraph(var2)) # use to call and make images b/c logistic loads slowly
                st.image('images7_7/logReg_'+var2+'.png') # use images made with previous line
        with col3:
                # st.components.v1.html(html_data,height=500)
                st.pyplot(get2VarScatter(var1, var2))#, ignore_streamlit_theme=True)   
        descriptiona = 'Description for ('+ var1.lower() + ', ' + var2.lower() +'):'
        st.write(descriptiona)
        st.latex(getLogisticInfo(var1, var2))                
        
with tab2:
        st.write('Summary statistics, logistic graph, scatterplot, and logistic formula information')
        col1a, col2a = st.columns([1,1])
        with col1a:
                # st.write('info1')
                ######
                # Select first customer feature
                var1a = st.selectbox(
                        'First customer feature ', colDict.keys() 
                )
                st.dataframe(getGraphSummary(var1a), use_container_width=True)
                var2a = st.selectbox(
                        'Second customer feature ', filter(lambda w:  w != var1, colDict.keys())
                )
                st.dataframe(getGraphSummary(var2a), use_container_width=True)
                
        with col2a:
                # st.write('===')
                # Select second customer feature 
                
                
                
                # altText for ['Number of products', 'Credit score', 'Age', 'Tenure', 'Balance', 'Estimated salary']
                st.write(getAltText(var1a, var2a))