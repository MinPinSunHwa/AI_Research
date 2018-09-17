# Wavelet Transformation----to denoise the input data

# imporing the relevant packages(need the PyWavelets as well)
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pylab as plt
from pywt import wavedec
from statsmodels.robust import mad
import seaborn
import numpy as np

#  1 layer Wavelet decomposition
x_data=x_data[:,0]                  # take one factor as an example (e.g Volumn, High, Hash rate) 
x_data = x_data.reshape(-1,1)       # reshape to make it easy to be preprocessed
x_scaler = MinMaxScaler()           # preprocessing the input data
x_data = x_scaler.fit_transform(x_data)
x_data = x_data.reshape(-1)         # reshape the data as (1,*) before decomposition

# decomposition
coeff = pywt.wavedec(x_data,'haar',level=2)

# wavelet denoising by means of thresholding
sigma = mad(coeff[-1])
uthresh =sigma*np.sqrt(2 * np.log(len(x_data)))  # calculates thresholding in the Wavelet domain
print(uthresh)

coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])  #Shrink coefficients by thresholding

#Reconstruct the signal
y = pywt.waverec(coeff, 'haar')


# plot the input and reconstruced input
plt.plot(list(range(len(x_data))),x_data)
plt.plot(list(range(len(x_data))),y,color='r')
plt.show()
