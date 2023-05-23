import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
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

##################################################################################################
# from viridis  https://waldyrious.net/viridis-palette-generator/
colors = ['rgb(189, 223, 38)', 'rgb(42, 120, 142)', 'rgb(72, 36, 117)']
layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
col1 = 'Periods'
col2 = 'Amount'
cols = (col1 + '', col2 + '', col1 + ' ', col2 + ' ', col1 + '  ', col2 + '  ', col1 + '   ', col2 + '   ')
colsFormat = {cols[0]: "{:.0f}".format, cols[1]: "{:.6f}".format, cols[2]: "{:.0f}".format, cols[3]: "{:.6f}".format, 
              cols[4]: "{:.0f}".format, cols[5]: "{:.6f}".format, cols[6]: "{:.0f}".format, cols[7]: "{:,.6f}".format}
# Initialize PV and FV to 1, rounding to 6 decimals, number of periods N is 0 to 20
# Initialize strings to print 
colorNum = 1
formType = 'FV'
# PV = 1
FV = 1
rnd = 6
N = np.arange(0, 21, 1)
TVMchoice = {
        'FV': ['Growth of a lump sum', 'FV', 'FV = PV \\times (1 + I)^N', colors[1], 20],
        'PV': ['Growth of a lump sum', 'PV', 'PV = \\frac{FV}{(1 + I)^N}', colors[2], 0],
        'FVann': ['Growth of an annuity', 'FV', 'FV = \\frac{PMT}{I} \\times ((1 + I)^N - 1)', colors[1], 20],
        'PVann': ['Growth of an annuity', 'PV', 'PV = \\frac{PMT}{I} \\times (1 - \\frac{1}{(1 + I)^N})', colors[2], 0],
}
# Makes bar graph of FV or PV of lump sum or annuity with formName = data type (FV, PV)  and (lump sum, annuity)
def getGraph(df1, formName, colorNum, N=20): 
        N = np.arange(0,N+1,1)
        fig = go.Figure(layout=layout, data=[go.Bar(name=TVMchoice[formName][0], x=N, y=df1.Dollars, marker_color = colors[colorNum])])
        # Change the bar mode
        deltax = 1
        if max(N) > 20: deltax = round(max(N)/20,0)
        fig.update_layout(
        title=TVMchoice[formName][0],
        xaxis = dict(
                title='Number of periods',
                tickmode = 'linear',
                tick0 = 0,
                dtick = deltax
        ),
        yaxis=dict(
                title='Amount ($)'
                # ,
                # titlefont_size=16,
                # tickfont_size=14,
        ),
        width=500,
        barmode="group", 
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', automargin=True)
        return fig
def getFormula(feat):
        if feat=='FV': LaTeXstr = 'FV = PV \\times (1 + I)^N'
        elif feat == 'PV': LaTeXstr = 'PV = \\frac{FV}{(1 + I)^N}'
        elif feat == 'FVann': LaTeXstr = 'FV = \\frac{PMT}{I} \\times \\left((1 + I)^N - 1\\right)'
        elif feat == 'PVann': LaTeXstr = 'PV = \\frac{PMT}{I} \\times \\left(1 - \\frac{1}{(1 + I)^N}\\right)'
        else: LaTeXstr = 'No formula'
        return LaTeXstr    

def roundVals(vals, rnd):
        return [round(n, rnd) for n in vals]

def getCalc(feat='FV', PV=1.00, FV=1.00, PMT=1.00, I=0.04, N=20):
        if feat=='FV': val = PV * (1 + I)**N
        elif feat == 'PV': val = FV/(1 + I)**N
        elif feat == 'FVann': val = PMT/I * ((1 + I)**N - 1)
        elif feat == 'PVann': val = PMT/I * (1 - 1/((1 + I)**N))
        else: val = "Incorrect paramater used." 
        return val
def getDataframe(feat='FV', PV=1.00, FV=1.00, PMT=1.00, I=0.04, N=20):
        N = np.arange(0,N+1,1)
        if feat=='FV': 
                
                val = PV * (1 + I)**N
        elif feat == 'PV': 
                PV = FV/((1 + I)**max(N))
                val = PV*(1 + I)**N
        elif feat == 'FVann': val = PMT/I * ((1 + I)**N - 1)
        elif feat == 'PVann': val = PMT/I * ((1 + I)**N - 1)
        else: val = N    
        data = {
                'Periods': N,
                'Dollars': roundVals(val,rnd)
        }
        df1 = pd.DataFrame(data)        
        return df1

def getTable(df1):
        rmax = len(df1)
        crange = 4
        rrange = max(1,int(math.ceil(float(rmax)/(crange))))
        print(str(rmax) +' = ' + str(rrange) + ' * ' + str(crange))
        dataTable = []
        for r in range(rrange):
                temp = []
                for c in range(crange):
                        indx = r + c*rrange
                        if indx < rmax:
                                temp.append(df1.iloc[indx,0]) 
                                temp.append(df1.iloc[indx,1]) 
                        else:
                                temp.append(float("nan")) 
                                temp.append(float("nan"))       
                dataTable.append(temp)
        dec = 2 # max(2,6 - int(max(df1.Dollars)/10.0))
        moneyFormat = "${:,."+ str(dec) + "f}"   
        colsFormat = {cols[0]: "{:.0f}".format, cols[1]: moneyFormat.format, cols[2]: "{:.0f}".format, cols[3]: moneyFormat.format, 
              cols[4]: "{:.0f}".format, cols[5]: moneyFormat.format, cols[6]: "{:.0f}".format, cols[7]: moneyFormat.format}
        dfTable = pd.DataFrame(
                dataTable,
                columns=cols).style.format(colsFormat, na_rep="---")
        return dfTable
def getSingPl(periods):
        if periods == 1: singPl = ''
        else: singPl = 's'
        return singPl
getSingPl(2)

# Initialize
df1 = getDataframe(feat='FV', PV=1.00, FV=1.00, PMT=1.00, I=0.04, N=20)
formula = getFormula(formType)

# Specify order of menu
config = {'displaylogo': False, 'displayModeBar': False}
# tab1, tab2 = st.tabs(['TVM plots', 'Description'])

col1, col2 = st.columns([1,3])
with col1:
        # Select TVM type to calculate
        FVorPV = st.selectbox(
                'FV or PV', ['FV','PV'] 
        )
        lumpOrAnnuity = st.selectbox(
                'Investment option', ['Lump sum', 'Annuity'] 
        )
        rate = st.number_input(
                "Interest rate (%)",
                min_value=0.00,
                step=0.25,
                value=4.00,
                key=11
        )
        periods = st.number_input(
                "Number of periods",
                min_value=1,
                step=1,
                value=20,
                key=12
        )
        
        if FVorPV == 'FV' and lumpOrAnnuity == 'Lump sum':
                Pval = st.number_input(
                        "Present value ($)",
                        min_value=0.00,
                        step=1.00,
                        value=3000.00,
                        key=13
                )
                formType = 'FV'
                df1 = getDataframe(feat='FV', PV=Pval, I=rate/100, N=periods)
                calc = getCalc(feat='FV', PV=Pval, I=rate/100, N=periods)
                formula = getFormula(formType)
                colorNum = 1
                latexSentence = '\\\\\\text{The future value in ' + str(periods) + ' period' + getSingPl(periods) +' of a lump sum investment of \\$'+ str("{:,.2f}".format(Pval)) +' at a rate of ' + str("{:.2f}".format(rate)) + '\\% is \\$' + str("{:,.2f}".format(calc)) + '.}'
                tableHeader = 'Lump sum investment amount' + getSingPl(periods) + ' at the end of N = ' + str(periods) + ' period' + getSingPl(periods) +' with PV = $'+ str(str("{:,.2f}".format(Pval))) + ' and I = ' + str("{:.2f}".format(rate))+ '% rate of return.'
        elif FVorPV == 'PV' and lumpOrAnnuity == 'Lump sum':
                Fval = st.number_input(
                        "Future value ($)",
                        min_value=0.00,
                        step=1.00,
                        value=5000.00,
                        key=13
                )
                formType = 'PV'
                df1 = getDataframe(feat='PV', FV=Fval, I=rate/100, N=periods)
                calc = getCalc(feat='PV', FV=Fval, I=rate/100, N=periods)
                formula = getFormula(formType)
                colorNum = 1
                latexSentence = '\\\\\\text{The present value of a lump sum investment with a future value of \\$'+ str("{:,.2f}".format(Fval)) +' in ' + str(periods) + ' period' + getSingPl(periods) +' at a rate of ' + str("{:.2f}".format(rate)) + '\\% is \\$' + str("{:,.2f}".format(calc)) + '.}'
                tableHeader = 'Lump sum investment amount' + getSingPl(periods) + ' at the end of N = ' + str(periods) + ' period' + getSingPl(periods) +' with FV = $'+ str(str("{:,.2f}".format(Fval))) + ' and I = ' + str("{:.2f}".format(rate))+ '% rate of return.'
        elif FVorPV == 'FV' and lumpOrAnnuity == 'Annuity':
                payment = st.number_input(
                        "Equal payment amount ($)",
                        min_value=0.00,
                        step=1.00,
                        value=2000.00,
                        key=13
                )
                formType = 'FVann'
                df1 = getDataframe(feat='FVann', PMT=payment, I=rate/100, N=periods)
                calc = getCalc(feat='FVann', PMT=payment, I=rate/100, N=periods)
                formula = getFormula(formType)
                colorNum = 1
                latexSentence = '\\\\\\text{The future value of an annuity when investing \\$'+ str("{:,.2f}".format(payment)) +' for ' + str(periods) + ' period' + getSingPl(periods) +' at a rate of ' + str("{:.2f}".format(rate)) + '\\% is \\$' + str("{:,.2f}".format(calc)) + '.}' 
                tableHeader = 'Investment amount' + getSingPl(periods) + ' for an annuity with PMT = $'+ str(str("{:.2f}".format(payment))) +' at the end of each of N = ' + str(periods) + ' period' + getSingPl(periods) + ' at I = ' + str("{:.2f}".format(rate))+ '%.'
        elif FVorPV == 'PV' and lumpOrAnnuity == 'Annuity':
                payment = st.number_input(
                        "Equal payment amount ($)",
                        min_value=0.00,
                        step=1.00,
                        value=2000.00,
                        key=13
                )
                formType = 'PVann'
                df1 = getDataframe(feat='PVann', PMT=payment, I=rate/100, N=periods)
                calc = getCalc(feat='PVann', PMT=payment, I=rate/100, N=periods)
                formula = getFormula(formType)
                colorNum = 1
                latexSentence = '\\\\\\text{The present value of an annuity when investing \\$'+ str(str("{:,.2f}".format(payment))) +' for ' + str(periods) + ' period' + getSingPl(periods) +' at a rate of ' + str("{:.2f}".format(rate)) + '\\% is \\$' + str("{:,.2f}".format(calc)) + '.}'
                tableHeader = 'Investment amount' + getSingPl(periods) + ' for an annuity with PMT = $'+ str(str("{:.2f}".format(payment))) +' at the end of each of N = ' + str(periods) + ' period' + getSingPl(periods) + ' at I = ' + str("{:.2f}".format(rate))+ '%.'
with col2:
        # Display graph
        st.plotly_chart(getGraph(df1, formType, colorNum, periods), config=config, ignore_streamlit_theme=True)
# Display TVM formula
st.latex(formula) 
st.latex(latexSentence)
check2 = st.checkbox("Show description")
if check2:
        st.write(tableHeader)
        st.table(getTable(df1,))


        

