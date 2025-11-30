import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("classification/iris.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("classification/iris_train.csv", index=False)
test_df.to_csv("classification/iris_test.csv", index=False)

df = pd.read_csv("regression/housing.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("regression/housing_train.csv", index=False)
test_df.to_csv("regression/housing_test.csv", index=False)