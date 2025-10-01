import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #0D98BA; 
        }
        [data-testid="stSidebar"] h1 {
            font-weight:bold; 
            font-size:36px;
            color:black;
        }
        [data-testid="stSidebar"] button {
            font-size: 18px;     
            padding: 12px 24px;      
            width: 200px;
            max-width:200px;
            min-width:200px;
            hover: blue; 
        }
        [data-testid="stSidebar"] button:hover {
            background-color: green; 
            color: #ffffff;
        }
            

    </style>
    """, unsafe_allow_html=True)



st.set_page_config(page_title="My Streamlit App", layout="wide", initial_sidebar_state="expanded")

df = pd.read_csv("Housing.csv")
df = pd.DataFrame(df)


st.sidebar.title("Sidebar Menu")
home = st.sidebar.button("Home")
data_table = st.sidebar.button("Data Table")
dashboard = st.sidebar.button("Dashboard")


if data_table:
    st.title("Data Table")
    search = st.text_input("Search")

    if search:
        filtered_data = df[
        df.apply(lambda row: search.lower() in row.astype(str).str.lower().to_string(), axis=1)
    ]
    else:
        filtered_data = df

    st.dataframe(filtered_data, use_container_width=True, height=500)

if dashboard:
    st.title("My Dashboard")
    col1, col2, col3 = st.columns(3)
    mean = df["price"].mean()
    
    with col1:
        st.metric(label="Mean Price", value="{:,}".format(round(df["price"].mean(), 2)))

    with col2:
        st.metric(label="STD Price", value="{:,}".format(round(df["price"].std(), 2)))

    with col3:
        st.metric(label="Median Price", value="{:,}".format(round(df["price"].median(), 2)))


    st.write("")
    st.write("")
    st.write("")

    fig, ax = plt.subplots(1, 2, figsize=(15,6))
    ax[0].hist(df["price"], bins=20)
    ax[0].set_title("Histogram of house price")
    ax[0].set_xlabel("Counts")
    ax[0].set_ylabel("Price")
    grouped = df.groupby("furnishingstatus")["price"].mean()
    ax[1].bar(grouped.index, grouped.values)
    ax[1].set_title("Average Price by Furnishing Status")
    ax[1].set_xlabel("Furnishing Status")
    ax[1].set_ylabel("Average Price")
    st.pyplot(fig)

    st.write("")
    st.write("")
    X = df[["area"]]
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)
    fig2, ax2 = plt.subplots(figsize=(15,6))
    ax2.scatter(X, y, color="blue", label="Data points")
    ax2.plot(X, model.predict(X), color="red", label="Regression line")
    ax2.set_title("House Price vs Area")
    ax2.set_ylabel("Price")
    ax2.set_xlabel("AreA")
    st.pyplot(fig2)

    intercept = model.intercept_
    coefficient = model.coef_[0]


    area = st.slider("Select area", min_value=1000, max_value=17000, step=500)
    st.write("")
    st.write(f"You selected: {area} sqm")

    result = coefficient * area + intercept
    st.write(f"The predicted price for {area} sqm is ${round(result):,} ")

    st.write("")
    plot_fig = px.scatter(df, x="area", y="price", color="furnishingstatus")
    st.plotly_chart(plot_fig)

    bar_fig = px.bar(
        df,
        x="furnishingstatus",
        y="price",
        title="Average Price by Furnishing Status",
        labels={"furnishingstatus": "Furnishing Status", "price": "Price"}
    )

    st.plotly_chart(bar_fig)

    hist_fig = px.histogram(
        df["price"],
        title="Histogram of house price" 

    )

    st.plotly_chart(hist_fig)
    df_grouped = df.groupby(['mainroad', 'furnishingstatus'], as_index=False)['price'].mean()
    clust_bar = px.bar(
        df_grouped,
        x="mainroad",
        y="price",
        color="furnishingstatus",
        barmode="group",
        title="Bar blot clustering"
    )
    st.plotly_chart(clust_bar)


