import pandas as pd
import plotly.express as px
import streamlit as st

df = pd.read_csv('/Users/shrikarstuff/Documents/MyOwnProjects/LungCancerML/runs/classify/train2/results.csv', header=0)
st.title("Train 2 Results: ")

df.rename(columns={'epoch': 'Epoch', '         train/loss': 'training_loss'}, inplace=True)

#epochs vs training loss: 
graph1 = px.line(
    df, 
    x='Epoch',
    y='training_loss',
    title = '<b>Epochs vs. training loss: </b>',
)
graph2 = px.line(
    df,
    x = 'Epoch',
    y = '               val/loss',
    title = '<b>Validation Accuracy vs Epochs: '
)

st.dataframe(df, use_container_width=True)
st.plotly_chart(graph1, use_container_width=True)
st.plotly_chart(graph2, use_container_width=True)


