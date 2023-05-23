import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
# import streamlit.components.v1 as components
# st.set_page_config(layout="wide")
# hides Streamlit footer and hamburger header 
# EX:  [data-testid=column]:nth-of-type(1) adjusts spacing within col1
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
        [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{gap: 2rem;}
        [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{gap: 1.2rem;}
        [data-testid=column]:nth-of-type(3) [data-testid=stVerticalBlock]{gap: 0rem; }
        }
        #root > div:nth-child(1) > div > div > div > div > section >
        div {padding-top: 1rem;}
        [role=radiogroup]{gap: 2.1rem;}
        div[data-testid="column"]:nth-of-type(1)
        {       text-align: end;}
        </style>
        '''
st.markdown(hide, unsafe_allow_html=True)

##################################################################################################
# from viridis  https://waldyrious.net/viridis-palette-generator/
colors = ['rgb(189, 223, 38)', 'rgb(42, 120, 142)', 'rgb(72, 36, 117)']
layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
col2 = 'Periods'
col3 = 'Amount'

fmtPer = "{:.2f}"
# fmtAmt = "{:.2f}"
PVinit = 1.0
PMTinit = 0.00
FVinit = 1.0
rateinit = 10.0
periodsinit = 1.0

cmpinit = 'Annually'
compound = ['Annually', 'Semiannually','quarterly','Monthly','Semimonthly','Bi-weekly','Weekly','Daily']
compoundNums = {'Annually': 1, 'Semiannually': 2,'quarterly': 4,'Monthly': 12,'Semimonthly': 24,'Bi-weekly': 26,'Weekly': 52,'Daily': 365}
cols = (col2 + '', col3 + '', col2 + ' ', col3 + ' ', col2 + '  ', col3 + '  ', col2 + '   ', col3 + '   ')
colsFormat = {cols[0]: fmtPer.format, cols[1]: "{:.6f}".format, cols[2]: fmtPer.format, cols[3]: "{:.6f}".format, 
              cols[4]: fmtPer.format, cols[5]: "{:.6f}".format, cols[6]: fmtPer.format, cols[7]: "{:,.6f}".format}
# Set session state variables
if 'key' not in st.session_state:
    st.session_state.key = 'value'

colorNum = 2
rnd = 6

# Makes bar graph of time values with given session state input
def getGraph(df1, colorNum, N=20.0): 
        feat = 'FV'
        N = np.arange(0.0,N+1.0,1.0, dtype=float)
        fig = go.Figure(layout=layout, data=[go.Bar(name=feat, x=N, y=df1.Dollars, marker_color = colors[colorNum])])
        # Change the bar mode
        deltax = 1
        if max(N) > 10: deltax = round(max(N)/10,0)
        # Session info
        cmp = st.session_state.cmpkey
        I = st.session_state.Ikey/100/compoundNums[cmp]
        I = "{:0.2f}".format(I*100) 
        ttl = 'Future values'
        fig.add_hline(y = 0)
        fig.update_layout(
                title=ttl,
                xaxis = dict(
                        title='Number of periods',
                        tickmode = 'linear',
                        tick0 = 0,
                        dtick = deltax,
                        tickfont_size=14,
                        titlefont_size=18
                ),
                yaxis=dict(
                        title='Amount',
                        titlefont_size=18,
                        tickfont_size=16
                ),
                width=300,
                barmode="group",
                yaxis_tickformat = '$' 
        )
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', automargin=True)
        return fig

def roundVals(vals, rnd):
        return [round(n, rnd) for n in vals]
def newtonRate(FV,PV,PMT,N):
        I = 5
        I2 = 1.1
        cnt = 0
        while cnt < 50: 
                cnt = cnt + 1
                den = (N*PV*(1+I)**(N-1)+PMT*(N*I*(1+I)**(N-1)-(1+I)**N +1)/(I**2))
                if (-.0000001< den) and (den < .0000001):
                        return 0
                else: 
                        I2 = I - (FV +PV*((1+I)**N) + PMT/I*((1+I)**N-1))/den
                        if abs(I2 - I) < 0.00001: 
                                # tf = False
                                return I2
                        else: 
                                I = I2
        return "\\text{Rate incalculable with inputs}"

def getPeriods(FV,PV,annI, PMT):
        result = 0
        try:
                result = (math.log(abs(-FV + PMT/annI))-math.log(abs(PV + PMT/annI)))/math.log(abs(1+annI))
        except:# OverflowError as err:
                result = "\\text{Periods incalculable with inputs}"
        return result


def getCalc():
        fmt = "\\${:,.2f}"
        cmp = st.session_state.cmpkey
        feat = st.session_state.opkey
        FV = st.session_state.FVkey
        PV = -st.session_state.PVkey
        I = st.session_state.Ikey/100/compoundNums[cmp]
        annI = st.session_state.Ikey/100
        PMT = -st.session_state.PMTkey
        N = st.session_state.Nkey
        if feat=='FV': 
                if I == 0: val = '\\text{FV} = ' + (fmt).format((PV + PMT*int(N)))
                else: val = '\\text{FV} = ' + (fmt).format((PV * (1 + I)**N + PMT/I * ((1 + I)**N - 1)))
        elif feat == 'PV':
                if I == 0: val = '\\text{PV} = ' + (fmt).format(-(FV  + PMT*int(N)))
                else: val = '\\text{PV} = ' + (fmt).format(-((FV + PMT/I)*((1 + I)**-N)- PMT/I))
        elif feat == 'Rate': 
                val = newtonRate(FV,-PV,-PMT,N)
                if val != "\\text{Rate incalculable with inputs}":
                        val = '\\text{I} = ' + '{:.2f}\%'.format(val*100)
        elif feat == 'PMT': val = '\\text{PMT} = ' + (fmt).format(-(FV-PV*(1 + I)**N)*I / ((1 + I)**N - 1))  
        elif feat == 'Periods': 
                val = getPeriods(FV,-PV,annI,-PMT)
                if val != "\\text{Periods incalculable with inputs}":
                        val = '\\text{N} = ' + ('{:,.2f}').format(val) + '\\text{ periods}'
        else: val = "Incorrect input"
        val.replace(',','{,}')
        return val
# Set data for future values to use in graph and table
def getDataframe():
        feat = st.session_state.opkey
        cmp = st.session_state.cmpkey
        feat = st.session_state.opkey
        FV = st.session_state.FVkey
        I = st.session_state.Ikey/100/compoundNums[cmp]
        PMT = -st.session_state.PMTkey
        N = st.session_state.Nkey
        PV = -st.session_state.PVkey
        if feat == 'PMT':
                PMT = (FV-PV*(1 + I)**N)*I / ((1 + I)**N - 1)
        if feat == 'PV': 
                if I == 0: PV = FV  + PMT*int(N)
                else: PV = (FV + PMT/I)*((1 + I)**-N)- PMT/I

        N = np.arange(0,st.session_state.Nkey+1.0,1.0,dtype=float)
        if I == 0: val = PV + PMT*int(N)
        else: val = PV * (1 + I)**N + PMT/I * ((1 + I)**N - 1)
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
        dec = 2 
        periodsFormat = "{:,.0f}" 
        moneyFormat = "${:,."+ str(dec) + "f}"   
        colsFormat = {cols[0]: periodsFormat.format, cols[1]: moneyFormat.format, cols[2]: periodsFormat.format, cols[3]: moneyFormat.format, 
              cols[4]: periodsFormat.format, cols[5]: moneyFormat.format, cols[6]: periodsFormat.format, cols[7]: moneyFormat.format}
        dfTable = pd.DataFrame(
                dataTable,
                columns=cols).style.format(colsFormat, na_rep="---")
        return dfTable

# Specify order of menu
config = {'displaylogo': False, 'displayModeBar': False}
# Set columns
col1, col2, col3, col4= st.columns([1,1.3,.6,2.1])
with col3:
        st.markdown('##')
        operation = st.radio('Calculation:', 
                ['PV', 'PMT', 'FV', 'Rate', 'Periods'], 
                key="opkey"
        )
# Set input names and calculation button
with col1:
        st.markdown('##')
        if operation == 'PV':
                st.text("CALCULATE BELOW")
        else:
                st.markdown("Present value (PV)")
        if operation == 'PMT':
                st.text("CALCULATE BELOW")
        else:
                st.markdown("Payments (PMT)")
        if operation == 'FV':
                st.text("CALCULATE BELOW")
        else:
                st.markdown("Future value (FV)")
        if operation == 'Rate':
                st.text("CALCULATE BELOW")
        else:
                st.markdown("Annual rate (I%)")
        if operation == 'Periods':
                st.text("CALCULATE BELOW")
        else:
                st.markdown("Periods (N)")
        st.markdown("Compounding")
        btn =  st.button("Calculate")
# Set input elements
with col2:      
        st.markdown('##')
        PV = st.number_input(
                "Present value (PV)",
                # min_value=0.00,
                step=1.00,
                value=PVinit,
                key='PVkey',
                label_visibility="collapsed"
        )

        PMT = st.number_input(
                "Payments (PMT)",
                # min_value=0.00,
                step=1.00,
                value=PMTinit,
                key='PMTkey',
                label_visibility="collapsed" 
        )
        FV = st.number_input(
                "Future value (FV)",
                # min_value=0.00,
                step=1.00,
                value=FVinit,
                key='FVkey',
                label_visibility="collapsed"
        )
        rate = st.number_input(
                "Annual rate (I %)",
                # min_value=0.00,
                step=0.250,
                value=rateinit,
                key='Ikey',
                label_visibility="collapsed"
        )     
        periods = st.number_input(
                "Number of periods",
                # min_value=1.0,
                step=1.0,
                value=periodsinit,
                key='Nkey',
                label_visibility="collapsed"
                
        )  
        # Select TVM type to calculate
        compounding = st.selectbox(
                'Compounding',  compound,
                key='cmpkey',
                label_visibility="collapsed" 
        )  
# Show calculation if btn is pressed
with col2:
        # Calculate value according to radio button session state 
        if btn: 
                calc = st.latex(getCalc())
        
# Initialize data with session state input
df1 = getDataframe()
with col4:
        check1 = st.checkbox("Show graph")
        if check1:
                # Display graph
                st.plotly_chart(getGraph(df1, colorNum, periods), config=config, ignore_streamlit_theme=True)   

check2 = st.checkbox("Show table")
if check2:
        st.write("Values over time")
        st.table(getTable(df1))
