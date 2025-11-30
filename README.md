# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
### Importing Necessary Libraries and Loading the Dataset
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#Load dataset
data=pd.read_csv('AirPassengers.csv')
```
### Declare required variables and set figure size, and visualise the data
```
N=1000
plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Mat
X=data['#Passengers']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
```
### Fitting the ARMA(1,1) model and deriving parameters
```
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
```
### Simulate ARMA(1,1) Process
```
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
```
### Plotting Simulated ARMA(2,2) Data
```
#Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
# Fitting the ARMA(1,1) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
#Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
#Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```
### OUTPUT:

<img width="830" height="443" alt="image" src="https://github.com/user-attachments/assets/82ae5aad-a7b6-4904-bbd5-449fc6fb5a13" />

Autocorrelation and Partial Autocorrelation:

<img width="828" height="414" alt="image" src="https://github.com/user-attachments/assets/d6c6fb15-94c6-46e9-bb50-310c60de3df8" />

SIMULATED ARMA(1,1) PROCESS:

<img width="829" height="434" alt="image" src="https://github.com/user-attachments/assets/6a451563-227e-4576-8b3a-13a2b9e92c89" />

Partial Autocorrelation

<img width="707" height="387" alt="image" src="https://github.com/user-attachments/assets/007f481f-15c6-465e-80c1-22f926aba152" />

Autocorrelation

<img width="711" height="379" alt="image" src="https://github.com/user-attachments/assets/87b86fbd-371c-4603-9472-6afccdbb5d3a" />

SIMULATED ARMA(2,2) PROCESS:

<img width="721" height="375" alt="image" src="https://github.com/user-attachments/assets/01e0294b-8d47-4673-a4f0-5ed219f34e58" />

Partial Autocorrelation

<img width="721" height="384" alt="image" src="https://github.com/user-attachments/assets/ab72b875-cad4-4bbd-aff7-797d5bfb6b1f" />

Autocorrelation

<img width="725" height="378" alt="image" src="https://github.com/user-attachments/assets/2700c97e-68a5-4ed7-9c89-9edaf72d4816" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
