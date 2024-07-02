import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# Carregar os dados
data = pd.read_csv('caminho_para_seus_dados.csv', index_col='Data', parse_dates=True)
print(data.head())
data.plot()
plt.show()
model = ARIMA(data, order=(5, 1, 0))  # (p,d,q) - Parâmetros do modelo ARIMA
model_fit = model.fit(disp=0)

print(model_fit.summary()) # Para poder ver o modelo

forecast, stderr, conf_int = model_fit.forecast(steps=10)
print(forecast)
plt.plot(data)
plt.plot(pd.date_range(start=data.index[-1], periods=10, freq='D'), forecast, color='red')
plt.fill_between(pd.date_range(start=data.index[-1], periods=10, freq='D'), conf_int[:, 0], conf_int[:, 1], color='pink')
plt.show()






