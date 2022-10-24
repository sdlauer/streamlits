import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# from st_aggrid import AgGrid, GridOptionsBuilder
# from st_aggrid.shared import GridUpdateMode
df = pd.read_csv(
    "diamonds_casestudy.csv"
)
# st.set_page_config(
#     layout="centered"
# )


# make pivot table
def pivotTable(val='price',indx='cut',cols='color',sortby='D'):
    return df.pivot_table(
            values=val,
            index=indx,
            columns=cols,
            aggfunc=np.size
            ).rename_axis(None,axis=1).reset_index().sort_values(
            by='D', ascending=False)
    # CSS to inject contained in a string



def descriptiveStats(num='price',cat='cut'):
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

    custom_dict = {'Fair': 4,'Good': 3,'Ideal': 0,'Premium': 1,'Very Good': 2}
    # self.num = num
    # self.cat = cat
    cat_cap = cat.capitalize()
    return df[[cat,num]].groupby(cat).agg(
        # Get mean of the numerical column for each group
        Mean=(num, np.mean),
        # Get min of the duration column for each group
        Median=(num, np.median),
        # Get sum of the duration column for each group
        Group_size=(num, np.size)).rename_axis(None,
            axis=1).reset_index().sort_values(by=[cat],
            key=lambda x: x.map(custom_dict)).rename(columns={cat:cat_cap})

# Hide row indices
num = 'price'
cat = 'cut'
catFeatures = ("cut", "color")
numFeatures = ["carat","depth","table","price","width","length","height"]
tab1, tab2 = st.tabs(["Summary statistics", "Plot"])
feature_dict ={
    "cut": ['Ideal','Premium','Very Good','Good','Fair'],
    "color": ['D','E','F','G','H','I','J']
    }
with tab2:
    col1, col2 = st.columns([1,4])

    with col1:
        plot = st.selectbox(
            "Plot",
            [
                "Box plot",
                "Density plot",
                "Histogram"
            ]
        )

        numerical = st.selectbox(
            "Numerical feature", numFeatures
        )
        genre = st.radio(
            "Categorical feature", catFeatures
        )

        categ = st.selectbox(
                'Select' + genre ,feature_dict[genre]
        )

    with col2:

        df1 = df[df[genre]==categ][numerical]
        fig, ax = plt.subplots()

        if plot == "Box plot":
            sns.boxplot(x=df1, width=0.5)

        elif plot == "Histogram":
            sns.histplot(x=df1)

        elif plot == "Density plot":
            sns.histplot(x=df1, kde=True, stat="density")

        ax.set_xlabel(numerical, fontsize=14)
        ax.ticklabel_format(style='plain', axis='x')

        if plot=="Histogram": ax.set_ylabel("Count", fontsize=14)
        if plot=="Density plot":
            ax.set_ylabel("Density", fontsize=14)
            ax.ticklabel_format(style='plain', axis='y')

        st.pyplot(fig)

with tab1:
        col1, col2 = st.columns([1,4])
        with col1:
            numerical1 = st.selectbox(
                "Numerical features", numFeatures
            )
            genre1 = st.radio(
                "Categorical features", catFeatures
            )
        with col2:
            st.subheader("Summary statistics for " + genre1)
            # pivotTable(val='price',indx='cut',cols='color',sortby='D')
            st.table(descriptiveStats(numerical1,genre1))
            # Display a static table

        # summary = df[df["categ"]==i].describe()
        # st.dataframe(summary)

# From Chris Chan df_manip.py
# st.header("Manipulating the df dataset")
# instructions = "Clicking Filters "
# col_ins = "Click Columns to display specific columns in the df dataset."
# row_ins = "Click Filters to display rows that satisfy a specific condition. "
# row_ex1 = "Ex: Clicking Population and selecting less than and typing 1000000 "
# row_ex2 = "returns rows where the population column is less than 1000000."
# piv1 = "Toggling Pivot Mode under Columns calculates various summary "
# piv2 = "statistics such as average population of countries in each categ."

# st.write(row_ins + row_ex1 + row_ex2)
# st.write(col_ins)
# st.write(piv1 + piv2)


# def aggrid_interactive_table(df: pd.DataFrame):
#     options = GridOptionsBuilder.from_dataframe(
#         df, enableRowGroup=True, enableValue=True, enablePivot=True
#     )
#
#     options.configure_side_bar()
#
#     options.configure_selection("single")
#     selection = AgGrid(
#         df,
#         enable_enterprise_modules=True,
#         height=400,
#         gridOptions=options.build(),
#         theme="alpine",
#         update_mode=GridUpdateMode.MODEL_CHANGED,
#     )
#
#     return selection




# selection = aggrid_interactive_table(df=df)

# st.table(pivotTable())
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
#
# hide = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         header {visibility: hidden;}
#         body {overflow: hidden;}
#         div.block-container {padding-top:1rem;}
#         div.block-container {padding-bottom:1rem;}
#         </style>
#         """
#
# st.markdown(hide, unsafe_allow_html=True)
#
# df = pd.read_csv("df_casestudy.csv")
# df.columns = ["carat","cut","color","clarity","depth","table","price","width","length","height"]
#
# col1, col2 = st.columns([2,3])
#
# with col1:
#     plot = st.table(df)
#
#
#     numerical = st.selectbox(
#         "Numerical feature",
#         [
#             "cut",
#             "price"
#         ]
#     )
#
#     categorical = st.selectbox(
#         "Categorical feature",
#         [
#             "cut",
#             "color"
#         ]
#     )
#
#     check = st.checkbox("Display summary statistics")
#
#     if check:
#         summary = df.groupby(categorical)[numerical].describe()
#         summary.columns = ["Count","Mean","Std", "Min", "Q1", "Median", "Q3", "Max"]
#         summary = summary[["Min", "Q1", "Median", "Q3", "Max"]]
#         st.dataframe(summary)
#
# with col2:
#     fig, ax = plt.subplots()
#
#     if plot == "Violin plot":
#         sns.violinplot(x=categorical, y=numerical, data = df)
#
#     elif plot == "Density plot":
#         sns.kdeplot(x=numerical, multiple="stack", hue=categorical, data = df)
#
#     elif plot == "Strip plot":
#         sns.stripplot(x=categorical, y=numerical, data = df)
#
#     elif plot == "Box plot":
#         sns.boxplot(x=categorical, y=numerical, data = df)
#
#     else:
#         sns.swarmplot(x=categorical, y=numerical, data = df)
#
#     if plot == "Density plot":
#         ax.set_xlabel(numerical, fontsize=14)
#         ax.set_ylabel("Density", fontsize=14)
#     else:
#         ax.set_xlabel(categorical, fontsize=14)
#         ax.set_ylabel(numerical, fontsize=14)
#
#     # st.pyplot(fig)
