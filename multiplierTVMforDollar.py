import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
N = np.arange(1, 21, 1)
TVMchoice = {
        'FV': ['FV lump sum multiplying factors for $1 PV', 'FV', 'FV = PV \\times (1 + I)^N', colors[1], 20],
        'PV': ['PV lump sum multiplying factors for $1 FV', 'PV', 'PV = \\frac{FV}{(1 + I)^N}', colors[2], 0],
        'FVann': ['FV annuity multiplying factors for $1 payment at the end of each period', 'FV', 'FV = \\frac{PMT}{I} \\times ((1 + I)^N - 1)', colors[1], 20],
        'PVann': ['PV annuity multiplying factors for $1 payment at the end of each period', 'PV', 'PV = \\frac{PMT}{I} \\times (1 - \\frac{1}{(1 + I)^N})', colors[2], 0],
}
# Makes bar graph of FV or PV of lump sum or annuity with formName = data type (FV, PV)  and (lump sum, annuity)
def getGraph(df1, formName, colorNum, I=4.00): 
        fig = go.Figure(layout=layout, data=[go.Bar(name=TVMchoice[formName][0], x=N, y=df1.Dollars, marker_color = colors[colorNum])])
        # Change the bar mode
        fig.update_layout(
        title=TVMchoice[formName][0],
        xaxis = dict(
                title='Number of periods',
                tickmode = 'linear',
                tick0 = 1,
                dtick = 1
        ),
        yaxis=dict(
                title='Dollars'
                # ,
                # titlefont_size=16,
                # tickfont_size=14,
        ),
        width=500,
        barmode="group", 
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
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
        elif feat == 'PV': val = FV/((1 + I)**N)
        elif feat == 'FVann': val = PMT/I * ((1 + I)**N - 1)
        elif feat == 'PVann': val = PMT/I * (1 - 1/((1 + I)**N))
        else: val = "Incorrect paramater used." 
        return val
def getDataframe(feat='FV', PV=1.00, FV=1.00, PMT=1.00, I=0.04):
        if feat=='FV': val = PV * (1 + I)**N
        elif feat == 'PV': val = FV/(1 + I)**N
        elif feat == 'FVann': val = PMT/I * ((1 + I)**N - 1)
        elif feat == 'PVann': val = PMT/I * (1 - 1/((1 + I)**N))
        else: val = N    
        data = {
                'Periods': N,
                'Dollars': roundVals(val,rnd)
        }
        df1 = pd.DataFrame(data)        
        return df1

def getTable(df1):
        
        dataTable = []
        for r in range(5):
                temp = []
                for c in range(4):
                        indx = r + c*5
                        temp.append(df1.iloc[indx,0]) 
                        temp.append(df1.iloc[indx,1])     
                dataTable.append(temp)
                
        dfTable = pd.DataFrame(
                dataTable,
                columns=cols).style.format(colsFormat, na_rep="---")
        return dfTable


# Initialize
df1 = getDataframe(feat='FV', PV=1.00, FV=1.00, PMT=1.00, I=0.04)
formula = getFormula(formType)

# Specify order of menu
config = {'displaylogo': False, 'displayModeBar': False}

col1, col2 = st.columns([1,3])
with col1:
        # Select TVM type to calculate
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

        if FVorPV == 'FV' and lumpOrAnnuity == 'Lump sum':
                formType = 'FV'
                df1 = getDataframe(feat=formType, I=rate/100)
                colorNum = 1

        elif FVorPV == 'PV' and lumpOrAnnuity == 'Lump sum':
                formType = 'PV'
                df1 = getDataframe(feat=formType, I=rate/100)
                colorNum = 2

        elif FVorPV == 'FV' and lumpOrAnnuity == 'Annuity':
                formType = 'FVann'
                df1 = getDataframe(feat=formType, I=rate/100)
                colorNum = 1

        elif FVorPV == 'PV' and lumpOrAnnuity == 'Annuity':
                formType = 'PVann'
                df1 = getDataframe(feat=formType, I=rate/100)
                colorNum = 2

with col2:
        # Display graph
        st.plotly_chart(getGraph(df1, formType, colorNum, rate), config=config, ignore_streamlit_theme=True)
# Display TVM formula
st.latex(getFormula(formType))
# Display multiplying factors for 20 periods
check2 = st.checkbox("Show description")
if check2:   
        st.write('Multiplying factors for ' + FVorPV + ' of ' + lumpOrAnnuity.lower() + ' at I = ' +str(rate) + '%:')   
        st.table(getTable(df1))


        

