import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

hide = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    body {overflow: hidden;}
    div.block-container {padding-top:1rem;}
    div.block-container {padding-bottom:1rem;}
    </style>
    """

st.markdown(hide, unsafe_allow_html=True)

col1, col2 = st.columns([2,3])

with col1:
    st.write('Binomial Distribution')
    nobs = st.number_input(
        "n",
        min_value=1,
        step=1,
        value=1,
        key=11
    )
    bprob = st.number_input(
        label="Probability of success",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
        key=12
    )

    calculate = st.selectbox(
        "Calculate",
        [
            "no calculation",
            "Probability given value",
 #           "Value given probability"
        ]
        )

    if calculate == "Probability given value":
        probability = st.selectbox(
            "Probability",
            [
                "P(X = value)",
                "P(X < = value)",
                "P(X > = value)",
                "P(value1 < = X < = value2)"
            ]
        )
        if probability == "P(value1 < = X < = value2)":
            value1 = st.number_input(
                "value1",
                min_value=0,
                max_value=int(nobs),
                step=1,
                value=0,
                key=21
            )
            value2 = st.number_input(
                "value2",
                min_value=0,
                max_value=int(nobs),
                step=1,
                value=1,
                key=22
            )
            
        else:
            value1 = st.number_input(
                "value",
                min_value=0,
                max_value=int(nobs),
                step=1,
                value=1,
                key=23
            )

with col2:
    if calculate == "Probability given value":
        fig, ax = plt.subplots()
        x = range(0, int(nobs)+1)
        ax.bar(x, height=binom.pmf(k=x, n=nobs, p=bprob), width=0.75, color='tab:blue')
        ax.set(xlabel='X', ylabel='Probability')
        title1=("binomial( %d, %0.2f)" %(nobs, bprob))
        ax.title.set_text(title1)
        plt.ylim(bottom=0)  
        if probability == "P(X < = value)":    
            x2 = range(0, int(value1)+1)
            ax.bar(x2, height=binom.pmf(k=x2, n=nobs, p=bprob), width=0.75, color='tab:orange')
            probcalc = binom.cdf(int(value1), n=nobs, p=bprob)
            text="P(X &le; %d) = %0.3f" %(value1, probcalc) 
        elif probability == "P(X > = value)":     
            x2 = range(int(value1), int(nobs)+1)
            ax.bar(x2, height=binom.pmf(k=x2, n=nobs, p=bprob), width=0.75, color='tab:orange')
            probcalc = 1-binom.cdf(int(value1)-1, n=nobs, p=bprob)
            text="P(X &ge; %d) = %0.3f" %(value1, probcalc)
        elif probability == "P(value1 < = X < = value2)": 
            if value1 < value2:    
                x2 = range(int(value1), int(value2)+1)
                y2 = binom.pmf(k=x2, n=nobs, p=bprob)
                ax.bar(x2, height=binom.pmf(k=x2, n=nobs, p=bprob), width=0.75, color='tab:orange')
                probcalc = binom.cdf(int(value2), n=nobs, p=bprob)-binom.cdf(int(value1)-1, n=nobs, p=bprob)
                text="P(%d &le; X &le; %d) = %0.3f" %(value1, value2, probcalc)
            else:
                fig, ax = plt.subplots()
                x = range(0, int(nobs)+1)
                ax.bar(x, height=binom.pmf(k=x, n=nobs, p=bprob), width=0.75, color='tab:blue')
                ax.set(xlabel='X', ylabel='Probability')
                title1=("binomial( %d, %0.2f)" %(nobs, bprob))
                ax.title.set_text(title1)
                plt.ylim(bottom=0)  
                text="value1 must be less than value2"               
        else:
            x2 = int(value1)
            y2 = binom.pmf(k=x2, n=nobs, p=bprob)
            ax.bar(x2, height=binom.pmf(k=x2, n=nobs, p=bprob), width=0.75, color='tab:orange')
            probcalc = binom.pmf(int(value1), n=nobs, p=bprob)
            text="P(X = %d) = %0.3f" %(value1, probcalc)        
    else:
        fig, ax = plt.subplots()
        x = range(0, int(nobs)+1)
        ax.bar(x, height=binom.pmf(k=x, n=nobs, p=bprob), color='tab:blue')
        ax.set(xlabel='X', ylabel='Probability')
        title1=("binomial( %d, %0.2f)" %(nobs, bprob))
        ax.title.set_text(title1)
        plt.ylim(bottom=0) 
        text=""
    st.pyplot(fig)
    st.markdown(f"""<div style='text-align: center'>{text}</h1>""", unsafe_allow_html=True)

    check2 = st.checkbox("Show description")
    if check2:
        mean1=nobs*bprob
        stdev1=(nobs*(bprob)*(1-bprob))**0.5
        alttext1= "Probability distribution graph of a %s distribution. The distribution \
        has a mean of %0.2f and a standard deviation of %0.2f" %(title1, mean1, stdev1)

        if calculate == "Probability given value":
            alttext2 = "The calculated probability is %s." %(text)
        else:
            alttext2=""
        st.write(alttext1)
        st.write(alttext2)

