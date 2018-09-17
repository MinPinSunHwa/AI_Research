### Get Data
btc_mean <- read.csv("D:/NeuralBC/AI_Model_Scripts/Data_Preprocessing/mean_price.csv", header = TRUE)
#btc_mean$Date <- as.Date(btc_mean$Date, format = "%m/%d/%y")
btc_mean$Date <- as.POSIXct(btc_mean$Date)
index <- with(btc_mean, order(Date))
Sorted_data <- btc_mean[index,]
head(btc_mean)
BTC_USD <- ts(Sorted_data$close)

library(forecast)
library(ggplot2)
library(tseries)

ggplot(btc_mean, aes(Date, close)) + geom_line() + ylab("USD") + xlab("")
dev.off()
### Clean Series ###
btc_mean$clean_close = tsclean(BTC_USD)
ggplot(btc_mean, aes(Date, close)) + geom_line() +  ylab("USD_cleaned") + xlab("")
dev.off()
### Moving Average ###
btc_mean$ma_close = ma(btc_mean$clean_close, order = 4)
btc_mean$ma30_close = ma(btc_mean$clean_close, order = 32)
ggplot() +
    geom_line(data = btc_mean, aes(x = Date, y = clean_close, colour = "USD")) +
    geom_line(data = btc_mean, aes(x = Date, y = ma_close, colour = "2 hours Moving")) +
    geom_line(data = btc_mean, aes(x = Date, y = ma30_close, colour = "16 hours Moving")) +
    ylab("USD")
dev.off()
### Decompose ########
MA_close <- ts(na.omit(btc_mean$ma_close), frequency = 30)
Decomp <- stl(MA_close, s.window = "periodic")
Deseasonal <- seasadj(Decomp)
#plot(Decomp)
autoplot(Decomp)
ggtsdisplay(Deseasonal)
dev.off()
############### Check whether data are stationary ##############
### KPSS test 

kpss.test(Deseasonal)
kpss.test(diff(Deseasonal))
tsdiag(model)
plot.ts(Deseasonal)

### ADF test
adf.test(Deseasonal, alternative = "stationary", k = 0)

### primary differentiation
ndiffs(log(BTC_USD)) # necessity for differentiation

par(mfrow = c(1, 2))
plot(log(Deseasonal), main = "original")
plot(diff(log(Deseasonal)), main = "differentiated")
dev.off()


## log transformation
plot.ts(log(Deseasonal))
################# Find optimized parameters p, d, q####################
########### Autocorrelations and choosing model order #################

### ACF and PACF
par(mfrow = c(1, 2))
#acf(log(Deseasonal), main = "ACF")
ggAcf(Deseasonal)
ggPacf(Deseasonal)
#pacf(log(Deseasonal), main = "PACF")
dev.off()

Fit_to_auto <- auto.arima(Deseasonal)
#plot(forecast(Fit_to_auto), h=4)
Fit_to_auto_diff <- auto.arima(diff(Deseasonal))
tsdiag(Fit_to_auto)
Fit_to_auto

################ Fit an ARIMA Model #####################################
Fit_to_arima <- arima(Deseasonal, c(2, 1, 2))
Fit_to_arima
autoplot(Fit_to_arima)
fcast <- forecast(Fit_to_arima, h = 140)
fcast
plot(fcast)