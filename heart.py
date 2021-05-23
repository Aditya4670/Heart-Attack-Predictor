import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("heart.csv")
# print(df.head())
# print(df.isnull().sum())

X = df.drop("target", axis=1)
Y = df["target"]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=40)
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.fit_transform(Xtest)
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print(model.score(Xtest, Ytest) * 100)
