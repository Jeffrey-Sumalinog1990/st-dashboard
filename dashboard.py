import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("Family Income and Expenditure.csv")

long_df = pd.melt(
    df,
    id_vars=["Region"],
    value_vars=[
        "Total Rice Expenditure",
        "Meat Expenditure",
        "Total Fish and  marine products Expenditure",
        "Fruit Expenditure",
        "Vegetables Expenditure"
    ],
    var_name="Food Category",
    value_name="Expenditure"
)

grouped = long_df.groupby("Food Category")["Expenditure"].mean().round(2)
grouped_df = grouped.reset_index()

st.subheader("Average Expenditure by Food Category")

fig = px.bar(
    grouped_df,
    x="Expenditure",
    y="Food Category",
    orientation="h",
    title="Average Household Expenditure by Food Category",
    labels={"Expenditure": "Average Expenditure", "Food Category": "Food Category"}
)
fig.update_layout(yaxis=dict(tickangle=0))

# Show plot
st.plotly_chart(fig)