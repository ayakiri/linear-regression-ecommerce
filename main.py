import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# prepare tools
df = pd.read_csv("Ecommerce Customers")

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 999)
sns.set()

# analyze data
sns.jointplot(df, x='Time on Website', y='Yearly Amount Spent')
sns.jointplot(df, x='Time on App', y='Yearly Amount Spent')
sns.lmplot(df, x='Yearly Amount Spent', y='Length of Membership')
plt.show()

# prepare data
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create model
model = LinearRegression()
model.fit(X_train, y_train)

# predict sales
predictions = model.predict(X_test)

# compare predictions and real values
plt.scatter(y_test, predictions)
plt.xlabel("Y Test")
plt.ylabel("Predicted Y values")

# evaluate model
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# analyze work - is it better to focus on developing website or mobile app?
coefficients = pd.DataFrame(model.coef_, X.columns)
coefficients.columns = ['Coeffecient']
print(coefficients)

# customers tend to spend more money on mobile app