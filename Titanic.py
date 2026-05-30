# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# loading the dataset

df=pd.read_csv("Titanic-Dataset.csv")



# %%
df

# %%
df.isnull().sum()

# %%
df.head(3)

# %%



# first we will see our data :-
df.info()
# we have datatype :-(int64 , float64)->{numerical data} (object)->{categorical data}


# cheaking the null values according to the columns
df.isnull().sum()
# printing some data abhout the dataset for understanding
print(df.columns)
print(f"rows=>{df.shape[0]}")
print(f"columns=>{df.shape[1]}")

# %%
# filling the null values 

# lets we make a function that fill the null values according to their type (float64,int64,object)

def fillnull(df):
    for col in df.columns:
        if(df[col].dtype=="int64" or df[col].dtype=="float64"):
            # filling the null values by mean:
            df[col].fillna(df[col].mean(),inplace=True)
        else:
            #filling by mode
            df[col].fillna(df[col].mode()[0],inplace=True) 
            


fillnull(df)



# %%
df.isnull().sum()

# %%
# we have cleaned all null values according to all their type:-
# sinse ml dont work on categorical columns so we have to make the categoriczal col to numnerical

categorical_columns=[]
for col in df.columns:
    if df[col].dtype=="object":
        categorical_columns.append(col)

print("the categorical columns are:", categorical_columns)


# lets see what kind of values these columns have
for col in categorical_columns:
    print(df[col].value_counts())


# from the analysis we can see that the columns sex and embarked have only two values so we can easily convert them to numerical by using map function or label encoder but the column ticket prefix have many values so we will drop it because it is not useful for our model .

# encoding..
df["Sex"]=df["Sex"].map({
    "male":0,
    "female":1
})
df["Embarked"]=df["Embarked"].map({
    "S":0,
    "C":1,
    "Q":2
})

# and now we drop the unuseful column






# %%
df.drop("Name",axis=1,inplace=True)

df.drop("Ticket",axis=1,inplace=True)

df.drop("Cabin",axis=1,inplace=True)

df.drop("PassengerId",axis=1,inplace=True)

# %%
# after encoding and dropping the unuseful columns we will see our data again


df

# %%
df.describe()

# %%
# importing the libraries for scaling the data
pip install scikit-learn

# %%
# Normalize/standardize the numerical features.

from sklearn.preprocessing import StandardScaler

num_cols = df.select_dtypes(include=["int64", "float64"]).columns


scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

print(df.head())



# %%

# lets see the outliers in the fare column by using boxplot
# we plot the boxplot for detecting the outliers


# the only valid column that can have outliers is fare 


import matplotlib.pyplot as plt
plt.boxplot(x="Fare" , data=df) 
plt.title("Fare Boxplot")
plt.ylabel("Fare")

plt.show()

# %%
# removing the outliers by using IQR method


Q1 = df["Fare"].quantile(0.25)

Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
# we should note that it will be lies in this range [lower,upper]
# apply the outliers only once becauese we loss data which will affect performance of the ml model 
df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)] 

# %%
# after removing the outliers we will see the boxplot again to make sure that we have removed the outliers
plt.boxplot(x="Fare" , data=df) 
plt.title("Fare Boxplot")
plt.ylabel("Fare")

plt.show()

# %%
# after removing outliers we see the data again
df.describe()

# %%



