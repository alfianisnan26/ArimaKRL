#!/usr/bin/env python
# coding: utf-8

# # ARPS Exponential Equation
# 
# persamaan ARPS dinyatakan dengan $$ Qt = Qi * (1 + bDt)^\frac{1}{b} $$
# 
# q = current production rate
# 
# q i = initial production rate (start of production)
# 
# d i = initial nominal decline rate at t = 0
# 
# t = cumulative time since start of production 
# 
# b= hyperbolic decline constant (0 < b < 1) 
# 
# $$ a = \frac{qt}{dq/dt} $$
# $$ D = \frac{1}{a} $$
# b formula is : $$ \frac{d\Bigl(\frac{qt}{(dqt/dt)}\Bigr)}{dt} $$

# In[67]:


import math
import pandas
import numpy as np
import matplotlib.pyplot as plt


# # DATA DARI 2014 - 2017 UNTUK PREDIKSI 2018

# In[68]:


#Pengambilan data dari csv
dd = pandas.read_csv('Kereta eR eL.csv', delimiter=';', index_col='Bulan')
nama_bulan=("Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember")


# In[69]:


print("Data original dari CSV\n")
print(dd)


# In[70]:


datad = dd['Pengguna'][0:48]
print("Data penguna KRL tahun 2014-2017\n")
print(datad)


# In[71]:


a=12.192573
b=352.15449
D1 = 1/a
output=[]
for t in range (49, 61):
    hasil = (((b*D1*t)+1)**(1/b))*datad[46]
    output.append(hasil)


# In[72]:


print("Hasil prediksi pengguna KRL tahun 2018 dari data tahun 2014 - 2017\n")
for i in range(len(output)):
    print(nama_bulan[i], end=': ')
    print(output[i])


# In[73]:


mse=[]
rata_mse=0
for i in range(len(output)):
    m=(output[i]-dd['Pengguna'][i+48])**2
    mse.append(m)
    rata_mse+=m


# In[74]:


print("Hasil Mean Square Error setiap bulan\n")
for i in range(len(output)):
    print(nama_bulan[i], end=': ')
    print(int(mse[i]))


# In[75]:


print("Hasil rata-rata MSE: ", end='')
print(int(rata_mse/12))


# In[76]:


df = pandas.read_csv('Kereta RL.csv', delimiter=';', index_col='Bulan')
x = (df['Pengguna'][11:24])
y = (output)
arps = y
x=x[1:len(x)]
plt.figure(figsize=(15,7))
plt.plot(nama_bulan, y, label='prediksi')
plt.plot(nama_bulan, x, label='data aktual')
plt.xlabel('Bulan')
plt.ylabel('Jumlah')
plt.title('Data pengguna KRL 2018')
plt.grid(True)
plt.legend()
plt.show()


# # DATA DARI 2017 UNTUK PREDIKSI 2018

# In[91]:


#Pengambilan data dari csv
data = df['Pengguna'][0:12]
print("Data original dari CSV\n")
print(df)


# In[95]:


#Pemrosesan ARPS
a = 107.435
D1 = 1/a
b = 224.9259
output=[]
for t in range (13, 25):
    hasil = (((b*D1*t)+1)**(1/b))*data[10]
    output.append(hasil)


# In[96]:


mse=[]
rata_mse=0
for i in range(len(output)):
    m=(output[i]-df['Pengguna'][i+12])**2
    mse.append(m)
    rata_mse+=m


# In[97]:


print("Data penguna KRL tahun 2017\n")
print(data)


# In[98]:


print("Data penguna KRL tahun 2018\n")
print(df['Pengguna'][12:24])


# In[99]:


print("Hasil prediksi pengguna KRL tahun 2018 dari data tahun 2017\n")
for i in range(len(output)):
    print(nama_bulan[i], end=': ')
    print(output[i])


# In[100]:


print("Hasil Mean Square Error setiap bulan\n")
for i in range(len(output)):
    print(nama_bulan[i], end=': ')
    print(int(mse[i]))


# In[101]:


print("Hasil rata-rata MSE: ", end='')
print(int(rata_mse/12))


# In[102]:


x = (df['Pengguna'][11:24])
y = (output)
x=x[1:len(x)]
plt.figure(figsize=(15,7))
plt.plot(nama_bulan, y, label='prediksi')
plt.plot(nama_bulan, x, label='data aktual')
plt.xlabel('Bulan')
plt.ylabel('Jumlah')
plt.title('Data pengguna KRL 2018')
plt.grid(True)
plt.legend()
plt.show()


# # PREDIKSI MENGGUNAKAN SARIMA

# In[103]:


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

data = sm.datasets.co2.load_pandas()
y = data.data

def parser(x):
    return pd.datetime.strptime('20'+x, '%Y-%m')
 
y = pd.read_csv('dataKRL.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)['2014-01-01':]

print(y)


# In[104]:


# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[105]:


#Menentukan order arima untuk membentuk AIC terkecil
warnings.filterwarnings("ignore") # specify to ignore warning messages

lowest=100000.0

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            if (lowest > results.aic):
                lowest = results.aic
                isLowest = 'ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, lowest)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
print('LOWEST AIC = ' + isLowest)


# In[106]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                               )

results = mod.fit()

print(results.summary().tables[1])


# In[107]:


results.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[108]:


pred = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=False)
pred_ci = pred.conf_int()
plt.figure(figsize=(15,8))
ax = y['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Tahun')
ax.set_ylabel('Penumpang (Ribu)')
plt.legend()

plt.show()


# In[109]:


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[110]:


pred_dynamic = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()
ax = y['2017':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2017-01-01'), y.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Tahun')
ax.set_ylabel('Penumpang (Ribu)')

plt.legend()
plt.show()


# In[111]:


# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = y['2017-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[112]:


pred_uc = results.get_forecast(steps=24)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Tahun')
ax.set_ylabel('Penumpang (Ribu)')

plt.legend()
plt.show()


# In[115]:


#pred_dynamic.predicted_mean.plot()
#y['2018-01-01':].plot()
#arps.data.plot()
plt.figure(figsize=(15,8))
plt.plot(nama_bulan, y['2018-01-01':], label='Data Aktual')
plt.plot(nama_bulan, pred_dynamic.predicted_mean, label='Prediksi ARIMA Dinamik')
plt.plot(nama_bulan, pred.predicted_mean, label='Prediksi ARIMA OSF')
plt.plot(nama_bulan, arps, label='Prediksi Arps')
plt.xlabel('Bulan')
plt.ylabel('Jumlah')
plt.title('Data pengguna KRL 2018')
plt.grid(True)
plt.legend()
plt.show()


# 
