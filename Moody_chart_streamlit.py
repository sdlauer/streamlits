import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

# st.set_page_config(layout="wide")
# hides Streamlit footer and hamburger header 
# NOTE   [data-testid=column]:nth-of-type(1) adjusts vertical spacing within col1
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
        # [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{gap: 3rem;}
        # [data-testid=column]:nth-of-type(3) [data-testid=stVerticalBlock]{gap: 0.3rem; }
        }
        #root > div:nth-child(1) > div > div > div > div > section >
        div {padding-top: 1rem;}
        # [role=radiogroup]{gap: 2.1rem;}
        div.row-widget.stRadio > div{flex-direction:row;}
        div[data-testid="column"]:nth-of-type(1)
        {       text-align: end;}
        </style>
        '''
st.markdown(hide, unsafe_allow_html=True)

##################################################################################################

def solve_colebrook(rel_roughness, reynolds_num):
    """Solve the Colebrook equation for the friction factor.

    Positional arguments:
    rel_roughness -- epsilon/D (right axis on graph)
    reynolds_num -- Re (bottom axis on graph)

    """
    def colebrook(f):
        return 1/np.sqrt(f) + 2 * np.log10(rel_roughness/3.7 + 2.51/(reynolds_num*np.sqrt(f)))
    return round(brentq(colebrook, 0.005, 0.1),4)
solve_colebrook_v = np.vectorize(solve_colebrook)
relative_roughness_values = [5E-2, 4E-2, 3E-2, 2E-2, 1.5E-2, 1E-2, 8E-3, 6E-3, 4E-3,
                                2E-3, 1E-3, 5E-4, 2E-4, 1E-4, 5E-5, 2E-5, 1E-5, 0]
# Makes Moody chart 
fig = plt.figure(figsize=(5,4))

## Couldn't get secondary axis to show in Streamlit 
# plt.ylabel( 'Relative roughness value')
xmin, xmax = 5E2, 1E8
ymin, ymax = 0.008, 0.1
n_pts = 100
# laminar and turbulent region
laminar_Re = np.linspace(1, 3500, n_pts)
turbulent_Re = np.logspace(np.log10(2000), np.log10(xmax), n_pts)
f_lam = 64 / laminar_Re
f_turb = np.zeros((n_pts, len(relative_roughness_values)))
for rough_ind, rough_val in enumerate(relative_roughness_values):
        f_turb[:, rough_ind] = solve_colebrook_v(rough_val, turbulent_Re)

plt.plot(laminar_Re, f_lam, label='f_lam')
labels = []
for relr in relative_roughness_values:
        labels.append("rgh "+str(relr))
plt.plot(turbulent_Re, f_turb, label=labels)

# Reynolds number labels (horizontal axis)
plt.xlim(xmin, xmax)
plt.xscale('log')
### LaTeX for graph labels doesn't currently work for pyplot in Streamlits
# plt.xlabel(r"Reynolds Number, $\mathit{Re}=\rho{}VD/\mu$")
# plt.ylabel(r"Friction Factor, $f$")   
### Secondary y-axis is not currently working, either
plt.xlabel("Reynolds Number, Re = rho V D / mu")
# Friction factor labels (left vertical axis)
plt.ylim(ymin, ymax)
plt.yscale('log')
plt.ylabel("Friction Factor f laminar") 
# transition region boundaries
trans_y = np.linspace(0,ymax,n_pts)
for i in range(0,100):
        trans_y[i] = '{:,.3f}'.format(trans_y[i])
trans_xmin = np.linspace(2000,2000,n_pts)
trans_xmax = np.linspace(3500,3500,n_pts)
plt.plot(trans_xmin, trans_y, color='grey', label='Trans_lower')
plt.plot(trans_xmax, trans_y, color='grey', label='Trans_upper')
# ax2.set_ylim(0,.1)
plt.gca().tick_params(which='both', right='off', top='off')
plt.gca().grid()


### Solved for Reynolds number in formula in Colebrook
def getReynolds(f,rel_roughness):
        re = abs(2.51/(pow(10, -0.5/np.sqrt(f))- rel_roughness/3.7)/np.sqrt(f))
        if re <640 : 
                return "\\text{No valid values within chart area}"
        else: 
                return re
def getGraph(op, reN, fturb, rel_rough):  
        # Plot points based on which radio button is selected and Reynolds number 
        lam = False
        if op == 'Reynolds Number--laminar' or op == 'Friction factor':
                lam = True
        if op == 'f_turb' and lam == False: 
                fturb = solve_colebrook(rel_rough, reN)
        if reN <= 3500 and (lam or op == 'No calculation'):
                plt.plot(reN, round(64/reN,4), marker='o', color='black', label='re,f_lam')
        # print(solve_colebrook_v(1E-3, turbulent_Re[50]))
        if reN >= 2000 and (lam == False or op == 'No calculation' or op == 'Friction factor'):
                plt.plot(reN, round(fturb,4), marker='o', color='black', label=str(rel_rough))
        return fig  
def expFormat(val): #, places):
        # fmt = '{:.'+str(places)+'f}'
        if val >= 10 and val<10000:
                num = "{:.0f}".format(val)
        elif val >10000:
                num = "{:.3e}".format(val)
        else:
                num = "{:.5e}".format(val)
                num = str(num.replace('-0','-').replace('+','').replace('e0','e'))
                num = num[:7] + '\\times 10^{' + num[8:] + '}'
        # num = round(val,3)      
        return str(num)
# Calcutions depending on which radio button is selected
def getCalc(op,reN,f_lam,f_turb,rel_roughness):      
        fmt = "{:.3f}"
        if op=='Reynolds Number--turbulent': 
                reN = getReynolds(f_turb,rel_roughness)
                if reN == '\\text{No valid values within chart area}':
                        val = reN
                        reN = 10e9 # no point on graph
                else: 
                        val = '\\text{Reynolds Number} = ' + expFormat(reN) + '\\text{ for }\\\\ \\text{ f\_turb = }' +expFormat(f_turb) +'\\text{, relative roughness = }' + fmt.format(rel_roughness)     
                        val += '\\\\ (\\text{but most likely not unique})'
        elif op=='Reynolds Number--laminar':
                maxf = 64/3500
                if f_lam > 0.1 or f_lam < maxf:
                        val = '\\text{In order to calculate a valid laminar Reynolds number,}'
                        val += '\\\\ \\text{enter a friction value between }' +expFormat(maxf)+  '\\text{ and 0.1}'
                        reN = 10e9 # no point on graph
                else: 
                        reN =  64/f_lam
                        val = '\\text{Reynolds Number} = ' + expFormat(reN) +'\\text{ for friction factor = }' + fmt.format(f_lam)
        elif op == 'f_turb':
                f_turb = solve_colebrook(rel_roughness, reN)
                val = '\\text{f\_turb} = ' + expFormat(f_turb) + '\\text{ for Reynolds Number = }' + expFormat(reN) +'\\text{, relative roughness = }' + fmt.format(rel_roughness) 
        elif op == 'Friction factor': 
                if reN <=3500 and reN >= 64/0.1:
                        f_lam = 64/reN
                        val = '\\text{f\_lam} = ' + fmt.format(f_lam) +'\\text{ for Reynolds number = }' + expFormat(reN)
                else:  
                        val ='\\text{Reynolds number must be between 635 and 3500 for a valid friction factor computation}'
        elif op == 'Relative roughness':
                if reN <2000: 
                        rel_roughness = 0.
                        val = '\\text{Relative roughness  = 0 for Reynolds number = }' + expFormat(reN)
                else:
                        # Solved for relative roughness in Colebrook formula
                        rel_roughness = (pow(10,-.5/np.sqrt(f_turb))  + 2.51/(reN*np.sqrt(f_turb)))*3.7
                        # rellist = getRelrough(f_turb,rel_roughness)
                        # if rellist == "Relative\ roughness\ not\ unique":
                        #         rellist = .09
                        val = '\\text{Relative roughness } = ' + fmt.format(rel_roughness) +'\\text{ is a valid estimate given}'
                        val += '\\\\ \\text{f\_turb = }' +expFormat(f_turb) +'\\text{, Reynolds number = }' + expFormat(re)
                        val += '\\\\ (\\text{but most likely not unique})'
        else: 
                val = '\\begin{align*}\\text{Number input: }\\\\'
                val += '&\\text{Reynolds number = }&' + expFormat(re) +'\\\\ &\\text{Relative roughness = }&' + fmt.format(rel_roughness)
                val += '\\\\ &\\text{f\_turb = }&' +expFormat(f_turb) 
                val += '\\\\ &\\text{Friction factor = }&' + fmt.format(f_lam) + '\\end{align*}'
        getGraph(op,reN,f_turb,rel_roughness)
        return val

# Initialize numerical input boxes
returbinit = 2500
relaminit = 2500
relroughinit = 0.02
flaminit = 64/relaminit
num = solve_colebrook(relroughinit, returbinit)
fturbinit = 0.06104488

# Specify order of menu
config = {'displaylogo': False, 'displayModeBar': False}

# Set columns
col1, col2 = st.columns([1,3])
# Set radio 
# with col2:

operation = st.radio('Select calculation:', 
        ['No calculation','Reynolds Number--turbulent', 'Reynolds Number--laminar', 'Relative roughness', 'f_turb', 'Friction factor'],                
        key="opkey"
)

# Set input names and calculation buttons
with col1:      
        st.markdown('Moody chart variables')
        re = st.number_input(
                "Reynolds Number",
                min_value=640., 
                max_value=1.e8,
                step=1e3,
                format="%1.2e",
                value=2500.,
                key='returbkey',
                # label_visibility="collapsed"
        )

        rel_rough = st.number_input(
                "Relative roughness",
                # min_value=0.00,
                min_value=0.00001,
                max_value=0.09,
                format="%.5f",
                step=.001,
                value=relroughinit,
                key='relroughkey',
                # label_visibility="collapsed"
        )
        f_turb = st.number_input(
                "f_turb",
                # min_value=0.00,
                min_value=0.008,
                max_value=0.085,
                format="%.5f",
                step=.001,
                value=fturbinit,
                key='fturbkey',
                # label_visibility="collapsed"
        )

        f_lam = st.number_input(
                "Friction factor",
                min_value=0.008,
                max_value=0.1,
                format="%.3f",
                step=.001,
                value=flaminit,
                key='flamkey',
                # label_visibility="collapsed" 
        )
btn =  st.button("Calculate")
        
# Show calculation if btn is pressed
if btn: 
        st.latex(getCalc(operation,re,f_lam,f_turb,rel_rough))
        
with col2:
        st.markdown('##')
        st.markdown('##')
        # Display graph
        st.plotly_chart(getGraph(operation,re,f_turb,rel_rough), theme="streamlit", config=config, ignore_streamlit_theme=True)   

