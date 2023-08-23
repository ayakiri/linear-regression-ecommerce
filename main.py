import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

print(X_train.columns)
print(model.coef_)
