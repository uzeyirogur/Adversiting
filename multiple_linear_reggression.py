import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv("data/Advertising.csv",index_col=0)
print(data.info())


#Çizim
#sns.pairplot(data)

#input
X = data[["TV","radio","newspaper"]]

#output
y = data["sales"].values.reshape(-1,1)

#sadece girdiler ve çıktılar çizimi
sns.pairplot(data,x_vars=data.columns[:3],y_vars=data.columns[3],height=5)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.30,random_state=0)

#Model oluşturma
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_predict = lr.predict(x_test)

#intercept , slope inceleme katsayıları yani betaları
print("incercept : ",lr.intercept_)
print("slope : ",lr.coef_)

katsayilar = pd.DataFrame(lr.coef_,columns=["beta1 TV","beta2 RADİO","beta3 NEWSPAPER"])

#Tahmin ve gerçek arasındaki değişim
indexler = range(1,61)
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(indexler,y_test,label="Grand Truth",color="red",linewidth=2)   #çizgi kalınlığı linewidth
ax.plot(indexler,lr_predict,label="Prediction",color="green",linewidth=2)
plt.title("SALES-ALL PREDİCTİON")
plt.xlabel("Data Index")
plt.ylabel("Sales")
plt.legend(loc="upper left")
plt.show()

#Hataları Çizelim : Residual (y - ^y) 
import numpy as np
indexler = range(1,61)
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(indexler,y_test - lr_predict,label="Residuals ",color="red",linewidth=2)   #çizgi kalınlığı linewidth
plt.title("SALES-ALL RESİDUALS")
plt.xlabel("Data Index")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.show()

#Sıfır doğrusu biz hatayı sıfır olsun isteriz
ax.plot(indexler,np.zeros(60),label="Zero Line ",color="black",linewidth=2)

#Doğruluk kontrol
from sklearn.metrics import r2_score,mean_squared_error
import math
r2 = r2_score(y_test, lr_predict)*100
mse = mean_squared_error(y_test, lr_predict)
rmse = math.sqrt(mse)


#OLS
import statsmodels.api as sm
x_train_ols = sm.add_constant(x_train)
sm_model = sm.OLS(y_train,x_train_ols)
result = sm_model.fit()
print(result.summary())

#newspaper p 0.05 ten büyük onu çıkart

sns.heatmap(data.corr(),annot=True)     #bu verilerin birbirleriyle ilişkisini anlatan tablo korelasyon(correlation)

#yeni model
x_train_new = x_train[["TV","radio"]]
x_test_new = x_test[["TV","radio"]]
lr.fit(x_train_new,y_train)
lr_newpredict = lr.predict(x_test_new)

r2_new = r2_score(y_test, lr_newpredict)*100
mse_new = mean_squared_error(y_test, lr_newpredict)
rmse_new = math.sqrt(mse_new)

x_train_ols_new = sm.add_constant(x_train_new)
sm_model = sm.OLS(y_train,x_train_ols_new)
result = sm_model.fit()
print(result.summary())

