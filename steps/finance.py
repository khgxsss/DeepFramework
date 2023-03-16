# Import required modules
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot

# Define the ticker symbol for Kakao Corp
tickerSymbol = '035720.KS'

# Fetch the data for Kakao Corp for the last 3 years
kakao = yf.Ticker(tickerSymbol)
kakao_data = kakao.history(period='3y')

# Create a pandas DataFrame with the closing prices
df = pd.DataFrame(kakao_data['Close'])

# Create an interactive plot using plotly
trace = go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Prices')
layout = go.Layout(title='Closing Prices of Kakao Corp', xaxis=dict(title='Date'), yaxis=dict(title='Price (KRW)'))
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)