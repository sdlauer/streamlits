import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from matplotlib.ticker import MaxNLocator

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
        [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{gap: .2rem;}
        [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{gap: 0rem; }
        }
        #root > div:nth-child(1) > div > div > div > div > section >
        div {padding-top: 1rem;}
        </style>
        '''
st.markdown(hide, unsafe_allow_html=True)
fig, ax = plt.subplots()
col1, col2 = st.columns([2,3])
with col1:
    st.write('Poisson Distribution')
    aveEvents = st.number_input(
        label='$\lambda$',
        min_value=1,
        step=1,
        value=8,
        key=11
    )
    title1=("Poisson distribution with mean $\lambda$ = %d" %(int(aveEvents)))
    ax.title.set_text(title1)
    calculate = st.selectbox(
        "Calculate",
        [
            "No calculation",
            "P(k = value)",
            "P(k ≤ value)",
            "P(k ≥ value)",
            "P(value1 ≤ k ≤ value2)"
        ]
        )
    if calculate == "P(value1 ≤ k ≤ value2)":
        value1 = st.number_input(
            "value1",
            min_value=0,
            step=1,
            value=0,
            key=21
        )
        value2 = st.number_input(
            "value2",
            min_value=0,
            step=1,
            value=1,
            key=22
        )
        
    elif calculate != "No calculation":
        value1 = st.number_input(
            "value",
            min_value=0,
            step=1,
            value=1,
            key=23
        )
with col2:
    x = range(0, int(aveEvents + 3 * aveEvents**0.5+1))
    ht = poisson.pmf(k=x, mu=int(aveEvents))
    ax.bar(x, height=ht, width=0.75, color='tab:blue')
    ax.set(xlabel='k', ylabel='Probability')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    text=''
    plt.ylim(bottom=0) 
    if calculate == "P(k ≤ value)":    
            x2 = range(0, value1+1)
            ax.bar(x2, height=poisson.pmf(k=x2, mu=int(aveEvents)), width=0.75, color='tab:orange')
            probcalc = poisson.cdf(k=int(value1), mu=int(aveEvents))
            text="P(k &leq; %d) = %0.3f" %(value1, probcalc) 
    elif calculate == "P(k ≥ value)":     
            x2 = range(int(value1), int(aveEvents + 3 * aveEvents**0.5+1))
            ax.bar(x2, height=poisson.pmf(k=x2, mu=int(aveEvents)), width=0.75, color='tab:orange')
            probcalc = 1-poisson.cdf(k=int(value1)-1, mu=int(aveEvents))
            text="P(k &geq; %d) = %0.3f" %(value1, probcalc)
    elif calculate == "P(value1 ≤ k ≤ value2)": 
            if value1 < value2:    
                x2 = range(int(value1), int(value2)+1)
                ax.bar(x2, height=poisson.pmf(k=x2, mu=int(aveEvents)), width=0.75, color='tab:orange')
                probcalc = poisson.cdf(k=int(value2), mu=int(aveEvents))-poisson.cdf(k=int(value1)-1, mu=int(aveEvents))
                text="P(%d &leq; k &leq; %d) = %0.3f" %(value1, value2, probcalc)
            else:
                # ax.bar(x, height=poisson.pmf(k=x, mu=int(aveEvents)), width=0.75, color='tab:blue')
                # ax.set(xlabel='k', ylabel='Probability')
                # plt.ylim(bottom=0)  
                text="value1 must be less than value2"               
    elif calculate == "P(k = value)":
            x2 = int(value1)
            ax.bar(x2, height=poisson.pmf(k=x2, mu=int(aveEvents)), width=0.75, color='tab:orange')
            probcalc = poisson.pmf(k=int(value1), mu=int(aveEvents))
            text="P(k = %d) = %0.3f" %(value1, probcalc)   
    st.pyplot(fig)
    st.markdown(f"""<div style='text-align: center'>{text}</div>""", unsafe_allow_html=True)

check2 = st.checkbox("Show description")
if check2:
    alttext1= "Probability distribution graph of a %s, so on average %d events happen with a standard deviation of %0.3f." %(title1, int(aveEvents), (aveEvents**0.5) )
    st.write(alttext1)
    if calculate == "Probability given value":
        alttext2 = "The calculated probability is %s." %(text)
        st.write(alttext2)  
    df = pd.DataFrame(poisson.pmf(k=x, mu=int(aveEvents)), columns=("P(k%s)" % i for i in ['']))
    df_T = df.transpose(copy=False)
    st.dataframe(df_T.style.format("{:.3f}"), column_config={"_index": "k"}, height=78)
    # st.write(st.__version__)