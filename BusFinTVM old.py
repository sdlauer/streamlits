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
calc = 0.0
fmtPer = "{:.2f}"
fmtAmt = "{:.2f}"
compound = ['Annually', 'Semiannually','quarterly','Monthly','Semimonthly','Bi-weekly','Weekly','Daily']
cols = (col1 + '', col2 + '', col1 + ' ', col2 + ' ', col1 + '  ', col2 + '  ', col1 + '   ', col2 + '   ')
colsFormat = {cols[0]: fmtPer.format, cols[1]: "{:.6f}".format, cols[2]: fmtPer.format, cols[3]: "{:.6f}".format, 
              cols[4]: fmtPer.format, cols[5]: "{:.6f}".format, cols[6]: fmtPer.format, cols[7]: "{:,.6f}".format}
# Initialize PV and FV to 1, rounding to 6 decimals, number of periods N is 0 to 20
# Initialize strings to print 
colorNum = 1
formType = 'FV'
# PV = 1
FV = 1
rnd = 6
N = np.arange(0.0, 21.0, 1.0, dtype=float)
TVMchoice = {
        'FV': ['Growth of a lump sum', 'FV', 'FV = PV \\times (1 + I)^N', colors[1], 20],
        'PV': ['Growth of a lump sum', 'PV', 'PV = \\frac{FV}{(1 + I)^N}', colors[2], 0],
        'FVann': ['Growth of an annuity', 'FV', 'FV = \\frac{PMT}{I} \\times ((1 + I)^N - 1)', colors[1], 20],
        'PVann': ['Growth of an annuity', 'PV', 'PV = \\frac{PMT}{I} \\times (1 - \\frac{1}{(1 + I)^N})', colors[2], 0],
}
if 'key' not in st.session_state:
    st.session_state.key = 'value'
# Makes bar graph of FV or PV of lump sum or annuity with formName = data type (FV, PV)  and (lump sum, annuity)
def getGraph(df1, formName, colorNum, N=20.0): 
        N = np.arange(0.0,N+1.0,1.0, dtype=float)
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
def newtonRate(FV,PV,PMT,N):
        print(FV,PV,PMT,N)
        I = 1
        I2 = 1.1
        for i in range(20):
                I2 = I - (-FV +PV*((1+I)**N) + PMT/I*((1+I)**N-1))/(N*PV*(1+I)**(N-1)+PMT*(N*I*(1+I)**(N-1)-(1+I)**N +1)/(I**2))
                if abs(I2 - I) < 0.00001: return I2
                else: I = I2
        return -404.04 
def newtonPMT(FV,PV,I,N):
        print(FV,PV,I,N)
        return -505.05

# def getCalc(feat='FV', PV=0.0, FV=0.0, PMT=0.0, I=0.0, N=0.0):
#         fmt = "${:,.2f}"
#         # if (PMT <=-0.0001 or PMT >= 0.0001):
#         #         feat = feat + 'ann'
#         if feat=='FV': 
#                 val = PV * (1 + I)**N + PMT/I * ((1 + I)**N - 1)
#                 st.session_state.FVkey = val
#                 # val = fmt.format(val)
#         elif feat == 'PV': 
#                 val = (FV - PMT/I * (1 - 1/((1 + I)**N)))/((1 + I)**N)
#                 st.session_state.PVkey = val  
#                 # val = fmt.format(val)
#         elif feat == 'Rate': 
#                 val = newtonRate(FV,PV,PMT,N)*100
#                 st.session_state.Ratekey = val 
#                 # val = "{:,.4f}%".format(val)
#         elif feat == 'PMT': 
#                 val = newtonPMT(FV,PV,I,N)*100
#                 st.session_state.PMTkey = val 
#                 # val = "{:,.4f}%".format(val)     
#         elif feat == 'Periods':
#                 val = 0.0
#                 st.session_state.Periodskey = val 
#                 # val = "{:,.2f} periods".format(val)
#         else: val = "Incorrect input"
#         print(val, st.session_state.FVkey, st.session_state.PVkey, st.session_state.PMTkey, st.session_state.Ratekey, st.session_state.Periodskey)
#         return val
def getCalc(feat):
        fmt = "${:,.2f}"
        # if (PMT <=-0.0001 or PMT >= 0.0001):
        #         feat = feat + 'ann'
        FV = st.session_state.FVkey
        PV = st.session_state.PVkey
        I = st.session_state.Ikey
        PMT = st.session_state.PMTkey
        N = st.session_state.Nkey

        if feat=='FV': 
                val = PV * (1 + I)**N + PMT/I * ((1 + I)**N - 1)
                # st.session_state.FVkey = val
                val = fmt.format(val)
        elif feat == 'PV': 
                val = (FV - PMT/I * (1 - 1/((1 + I)**N)))/((1 + I)**N)
                # st.session_state.PVkey = val  
                val = fmt.format(val)
        elif feat == 'Rate': 
                val = newtonRate(FV,PV,PMT,N)*100
                # st.session_state.Ikey = val 
                val = "{:,.4f}%".format(val)
        elif feat == 'PMT': 
                val = newtonPMT(FV,PV,I,N)*100
                # st.session_state.PMTkey = val 
                val = "{:,.4f}%".format(val)     
        elif feat == 'Periods':
                val = 0.0
                # st.session_state.Nkey = val 
                val = "{:,.2f} periods".format(val)
        else: val = "Incorrect input"
        print(val, st.session_state.FVkey, st.session_state.PVkey, st.session_state.PMTkey, st.session_state.Ratekey, st.session_state.Periodskey)
        return val
def getDataframe(feat='FV', PV=1.00, FV=1.00, PMT=1.00, I=0.04, N=20.0):
        N = np.arange(0,N+1.0,1.0,dtype=float)
        # print(N)
        if feat=='FV':                
                val = PV * (1 + I)**N
        elif feat == 'PV': 
                PV = FV/((1 + I)**max(N))
                val = PV*(1 + I)**N
        elif feat == 'FVann': val = PMT/I * ((1 + I)**N - 1)
        elif feat == 'PVann': val = PMT/I * ((1 + I)**N - 1)
        elif feat == 'PMT': val = (FV * I)/((1+I)**N -1)
        # elif feat == 'Rate': val = newtonRate(I)
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
        # print(str(rmax) +' = ' + str(rrange) + ' * ' + str(crange))
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
        periodsFormat = "{:,.0f}" 
        moneyFormat = "${:,."+ str(dec) + "f}"   
        colsFormat = {cols[0]: periodsFormat.format, cols[1]: moneyFormat.format, cols[2]: periodsFormat.format, cols[3]: moneyFormat.format, 
              cols[4]: periodsFormat.format, cols[5]: moneyFormat.format, cols[6]: periodsFormat.format, cols[7]: moneyFormat.format}
        dfTable = pd.DataFrame(
                dataTable,
                columns=cols).style.format(colsFormat, na_rep="---")
        return dfTable
def getSingPl(periods):
        if periods <= 1.00001 and periods <= 0.9999: singPl = ''
        else: singPl = 's'
        return singPl
getSingPl(2)
inits = [0.0,0.0,0.0,0.0,0.0,] 
# def capture(PVinit=0.0, FVinit=0.0, PMTinit=0.0, rateinit=0.0, periodsinit=0.0):

# Initialize
df1 = getDataframe(feat='FV', PV=1.00, FV=1.00, PMT=1.00, I=0.04, N=20.0)
formula = getFormula(formType)

# Specify order of menu
config = {'displaylogo': False, 'displayModeBar': False}
operation = 'FV'
periods = 20.0
PVinit = 1000.0
PMTinit = 0.00
FVinit = 3000.0
rateinit = 5.0
periodsinit = 10.0
col1, col2,col3= st.columns([2,2,5])
with col2:
        # Select TVM type to calculate
        compounding = st.selectbox(
                'Compounding',  compound
        )
        st.markdown('##')
        operation = st.radio('Select Calculation', ['PV', 'PMT', 'FV', 'Rate', 'Periods'])

        st.markdown('##')
        # st.markdown(operation + " calculation")

with col1:
# if btn:
#         PVinit = calc
# if operation == 'PV':
#         # PV = 0.0
#         st.info("Calculating present value")
        
# else:
        PV = st.number_input(
        "Present value (PV)",
        min_value=0.00,
        step=1.00,
        # value=PVinit,
        key='PVkey'
        )

# if operation == 'PMT':
#         PMT = 0.0
#         st.info("Calculating payments")
# # elif operation == 'Rate':
# #         st.info("Calculating rate of return") 
# else:
        PMT = st.number_input(
                "Payments (PMT)",
                min_value=0.00,
                step=1.00,
                # value=PMTinit,
                key='PMTkey', 
        )
# if operation == 'FV':
#         FV = 0.0
#         st.info("Calculating future value")
# else:
        FV = st.number_input(
                "Future value (FV)",
                min_value=0.00,
                step=1.00,
                # value=FVinit,
                key='FVkey',
        )
# if operation == 'Rate':
#         rate = 0.0
#         st.info("Calculating rate of return")           
# else:
        rate = st.number_input(
                "Annual rate (I %)",
                # min_value=0.00,
                step=0.25,
                # value=rateinit,
                key='Ikey',
        )  
# if operation == 'Periods':
#         periods = 0.0
#         st.info("Calculating periods")
# else:        
        periods = st.number_input(
                "Number of periods",
                min_value=0.0,
                step=1.0,
                # value=periodsinit,
                key='Nkey',
                
        )             

with col2:
        
        
        # lumpOrAnnuity ="no"

        
        
        # if operation == 'FV':
        #         # PV = st.number_input(
        #         #         "Present value ($)",
        #         #         min_value=0.00,
        #         #         step=1.00,
        #         #         value=3000.00,
        #         #         key=13,
        #         # )
        #         formType = 'FV'
        #         df1 = getDataframe(feat='FV', PV=PV, I=rate/100, N=periods)
        #         calc = getCalc(operation)
        #         formula = getFormula(formType)
        #         colorNum = 1
        #         latexSentence = '\\\\\\text{The future value in ' + str(periods) + ' period' + getSingPl(periods) +' of a lump sum investment of \\$'+ str("{:,.2f}".format(PV)) +' at a rate of ' + str("{:.2f}".format(rate)) + '\\% is ' + calc + '.}'
        #         tableHeader = 'Lump sum investment amount' + getSingPl(periods) + ' at the end of N = ' + str(periods) + ' period' + getSingPl(periods) +' with PV = $'+ str(str("{:,.2f}".format(PV))) + ' and I = ' + str("{:.2f}".format(rate))+ '% rate of return.'
        # elif operation == 'PV':
        #         # Fval = st.number_input(
        #         #         "Future value ($)",
        #         #         # min_value=0.00,
        #         #         step=1.00,
        #         #         value=5000.00,
        #         #         key=13
        #         # )
        #         formType = 'PV'
        #         df1 = getDataframe(feat=operation, FV=FV, I=rate/100, N=periods)
        #         calc = getCalc(operation)
        #         formula = getFormula(formType)
        #         colorNum = 1
        #         PVinit = calc
        #         latexSentence = '\\\\\\text{The present value of a lump sum investment with a future value of \\$'+ str("{:,.2f}".format(FV)) +' in ' + str(periods) + ' period' + getSingPl(periods) +' at a rate of ' + str("{:.2f}".format(rate)) + '\\% is ' + calc + '.}'
        #         tableHeader = 'Lump sum investment amount' + getSingPl(periods) + ' at the end of N = ' + str(periods) + ' period' + getSingPl(periods) +' with FV = $'+ str(str("{:,.2f}".format(FV))) + ' and I = ' + str("{:.2f}".format(rate))+ '% rate of return.'
        # elif operation == 'FV' and lumpOrAnnuity == 'Annuity':
        #         payment = st.number_input(
        #                 "Equal payment amount ($)",
        #                 # min_value=0.00,
        #                 step=1.00,
        #                 value=2000.00,
        #                 key=13
        #         )
        #         formType = 'FVann'
        #         df1 = getDataframe(feat='FVann', PMT=payment, I=rate/100, N=periods)
        #         # calc = getCalc(feat='FVann', PMT=payment, I=rate/100, N=periods)
        #         formula = getFormula(formType)
        #         colorNum = 1
        #         latexSentence = '\\\\\\text{The future value of an annuity when investing \\$'+ str("{:,.2f}".format(payment)) +' for ' + str(periods) + ' period' + getSingPl(periods) +' at a rate of ' + str("{:.2f}".format(rate)) + '\\% is ' + calc + '.}' 
        #         tableHeader = 'Investment amount' + getSingPl(periods) + ' for an annuity with PMT = $'+ str(str("{:.2f}".format(payment))) +' at the end of each of N = ' + str(periods) + ' period' + getSingPl(periods) + ' at I = ' + str("{:.2f}".format(rate))+ '%.'
        # elif operation == 'PV' and lumpOrAnnuity == 'Annuity':
        #         payment = st.number_input(
        #                 "Equal payment amount ($)",
        #                 # min_value=0.00,
        #                 step=1.00,
        #                 value=2000.00,
        #                 key=13
        #         )
        #         formType = 'PVann'
        #         df1 = getDataframe(feat='PVann', PMT=payment, I=rate/100, N=periods)
        #         calc = getCalc(feat='PVann')
        #         formula = getFormula(formType)
        #         colorNum = 1
        #         latexSentence = '\\\\\\text{The present value of an annuity when investing \\$'+ str(str("{:,.2f}".format(payment))) +' for ' + str(periods) + ' period' + getSingPl(periods) +' at a rate of ' + str("{:.2f}".format(rate)) + '\\% is ' + calc + '.}'
        #         tableHeader = 'Investment amount' + getSingPl(periods) + ' for an annuity with PMT = $'+ str(str("{:.2f}".format(payment))) +' at the end of each of N = ' + str(periods) + ' period' + getSingPl(periods) + ' at I = ' + str("{:.2f}".format(rate))+ '%.'
        btn =  st.button("Calculate") 
        if btn: 
                calc = st.info(getCalc(operation))
  

with col3:
        # Display graph
        st.plotly_chart(getGraph(df1, formType, colorNum, periods), config=config, ignore_streamlit_theme=True)   
        # Display TVM formula
        # st.latex(formula) 


# DEFAULT_NUMBER = 0.0

# if "example" not in st.session_state:
#     st.session_state.example = DEFAULT_NUMBER

# placeholder = st.empty()

# if st.button('Reset'):
#     st.session_state.example = DEFAULT_NUMBER

# example = placeholder.number_input('Example', min_value=0.0, max_value=1.0,step=0.1, format='%.7f', key="example")

# st.markdown("NOTE:  PV and FV are opposite in sign.  FV and PMT have the same sign.")
# st.latex(latexSentence)
check2 = st.checkbox("Show description")
if check2:
        # st.write(tableHeader)
        st.table(getTable(df1,))


import datetime

st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.last_updated = datetime.time(0,0)

def update_counter():
    st.session_state.count += st.session_state.increment_value
    st.session_state.last_updated = st.session_state.update_time

with st.form(key='my_form'):
    st.time_input(label='Enter the time', value=datetime.datetime.now().time(), key='update_time')
    st.number_input('Enter a value', value=0, step=1, key='increment_value')
    submit = st.form_submit_button(label='Update', on_click=update_counter)

st.write('Current Count = ', st.session_state.count)
st.write('Last Updated = ', st.session_state.last_updated)

