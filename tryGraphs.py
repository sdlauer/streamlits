# import plotly.express as px
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
# import plotly
# from itertools import cycle
# from viridis  https://waldyrious.net/viridis-palette-generator/
colors = ['rgb(189, 223, 38)', 'rgb(42, 120, 142)', 'rgb(72, 36, 117)']
layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
# df = px.data.iris()
 
# fig = px.bar(df, x="sepal_width", y="sepal_length", color="species",
#             hover_data=['petal_width'], barmode = 'stack')
 
# fig.show()
# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# plt.scatter(x, y)
# plt.show()


# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }

# df1 = pd.DataFrame(data, index = ["1", "2", "3"])
# fig = px.bar(df1, x="calories", y="duration",
#              barmode = 'stack')
# fig.show()
# print(df1) 

def getFVlumpSum(PV, I, N):
    return PV * (1 + I)**N

def getPVlumpSum(FV,I,N):
    return FV/(1 + I)**N

def getFVannuity(PMT, I, N):
    return PMT/I * ((1 + I)**N - 1)

def getPVannuity(PMT, I, N):
    return PMT/I * (1 - 1/((1 + I)**N))

def getPVannuityDue(PMT, I, N):
    return getPVannuity(PMT, I, N)*1+I

def getN(FV,PV,I):
    return [math.log(FV/PV)/math.log(1 + i) for i in I]
def roundVals(rnd, vals):
    return [round(n, rnd) for n in vals]

# PV = 1
# I = 5/100
# N = np.arange(0, 21, 1)
# FV= getPVlumpSum(PV, I, N)
# # x = np.array([-250000, 100000, 150000, 200000, 250000, 300000])
# # r = 2*x
# rnd = 2
# values = roundVals(rnd, FV)
# # data = {
# #     "Periods": N, 
# #     "Dollars": values
# # }

# df1 = pd.DataFrame(data)
# fig = px.bar(df1, x="Periods", y="Dollars",
#              barmode = 'stack')
# fig.show()
# print(df1) 
PV = 1
FV = 2*PV
rnd = 2
I = np.arange(1.0, 21.0, 1)

N = roundVals(rnd, getN(FV,PV,I/100))
data = {
    "Rate": I,
    "Actual_Periods": roundVals(rnd, N), 
    "Rule72": roundVals(rnd, 72/I),
    "Rule70": roundVals(rnd, 70/I)
}
df1 = pd.DataFrame(data)
# fig = px.bar(df1, x="Periods", y="Dollars",
#              barmode = 'group')
# fig.show()

# dftable = "\n".join("{} {} {} {} {} {} {} {}".format(r1, per1,r2, per2,r3, per3,r4, per4) for r1, per1,r2, per2,
#                 r3, per3,r4, per4 in zip(df1.Rate[0:5], df1.Actual_Periods[0:5], df1.Rate[5:10], df1.Actual_Periods[5:10], df1.Rate[10:15], df1.Actual_Periods[10:15], df1.Rate[15:20], df1.Actual_Periods[15:20]))
# dftable = pd.DataFrame(dftable)
#     {"Rate1": df1.Rate[0:5], 'Period1': df1.Actual_Periods[0:5], 
#     "Rate2": df1.Rate[5:10], 'Period1': df1.Actual_Periods[5:10], 
#     "Rate3": df1.Rate[10:15], 'Period1': df1.Actual_Periods[10:15], 
#     "Rate4": df1.Rate[15:20], 'Period1': df1.Actual_Periods[15:20]}
# )
features = {
        'FV': ['FV of a lump sum', 'FV', 'PV \\times (1 + I)^N'],
        'PV': ['PV of a lump sum', 'PV', '\\frac{FV}{(1 + I)^N}'],
        'PVann': ['FV of an annuity', 'FV', '\\frac{PMT}{I} \\times ((1 + I)^N - 1)'],
        'FVann': ['PV of an annuity', 'PV', '\\frac{PMT}{I} \\times (1 - \\frac{1}{(1 + I)^N})'],
}
print("   Rate  Periods      Rate  Periods      Rate  Periods      Rate  Periods")
for r1, per1,r2, per2, r3, per3,r4, per4 in zip(df1.Rate[0:5], df1.Actual_Periods[0:5], 
                                                df1.Rate[5:10], df1.Actual_Periods[5:10], 
                                                df1.Rate[10:15], df1.Actual_Periods[10:15], 
                                                df1.Rate[15:20], df1.Actual_Periods[15:20]):
    
    print(f'{r1:7.2f} {per1:7.2f} {r2:10.2f} {per2:7.2f} {r3:10.2f} {per3:7.2f} {r4:10.2f} {per4:7.2f}') 
# print(dftable) 

print(features['FVann'][2])

groups=['Actual_Periods', 'Rule72', 'Rule70']

fig = go.Figure(layout=layout,
    data=[
        go.Bar(name='Actual periods', x=I, y=df1.Actual_Periods, marker_color = colors[2]),
        go.Bar(name='Rule of 72 estimate', x=I, y=df1.Rule72, marker_color = colors[0]),
        go.Bar(name='Rule of 72 estimate', x=I, y=df1.Rule70, marker_color = colors[1])
        ]
    )
# Change the bar mode
fig.update_layout(
    title='The time it takes for principal to double in value',
    xaxis = dict(
        title='Rate (%)',
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    yaxis=dict(
        title='Years'
        # ,
        # titlefont_size=16,
        # tickfont_size=14,
    ),
    barmode="group", 
    )
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.show()