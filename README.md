# DATASET EXPLANATION ON MARVEL CHARACTERS

## Program developed by :

         Yuvadarshini S (212221230126)
         Mirudhula D (212221230060)
         Dharshini D S (212221230022)


## BIG DATA ANALYTICS:


         This article utilizes statistical methods, data mining techniques and Python in order to create a prediction model for superhero characters alignment.

         Starting off we choose what dataset to utilize — which is then prepared, handled and analyzed. 
         
         Then, the cleansed data is imported to create a viable prediction model that can foretell the alignment based on descriptions of the characters. 
         
         This outcome is then tested and debated to describe the different results.
         
## Program:
```
import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv("MarvelMovies.csv")
df

df.head()

df.info()

df.isnull()

df.isnull().sum()

df['FIRST APPEARANCE']=df['FIRST APPEARANCE'].fillna(df['GENDER'].mode()[0])
df['GENDER']=df['GENDER'].fillna(df['GENDER'].mode()[0])
df['ALIGN']=df['ALIGN'].fillna(df['GENDER'].mode()[0])

df['YEAR']=df['YEAR'].fillna(df['APPEARANCES'].mean())
df['APPEARANCES']=df['APPEARANCES'].fillna(df['APPEARANCES'].mean())
df.head()

df['YEAR']=df['YEAR'].fillna(df['YEAR'].median())

df.info()

df.head()

df.shape()

df.describe()

df.tail()

df.columns

df.duplicated()

df.dtypes

df['APPEARANCES'].value_counts()

sns.boxplot(x="APPEARANCES",data=df)

sns.countplot(x='ID',data=df)


sns.distplot(df["YEAR"])

sns.scatterplot(df['APPEARANCES'],df['HAIR'])


plt.xticks(rotation = 90)
sns.barplot(df['GENDER'],df['YEAR'],hue=df['ALIVE'])



states=df.loc[:,["HAIR","YEAR"]]
states=states.groupby(by=["HAIR"]).sum().sort_values(by="YEAR")
#plt.figure(figsize=(10,7))
sns.barplot(x=states.index,y="YEAR",data=states)
plt.xticks(rotation = 90)
plt.xlabel=("APPEARANCES")
plt.ylabel=("COUNT")
plt.show()

import matplotlib.pyplot as plt
plt.plot(df.YEAR, df.APPEARANCES)

sns.countplot(x="HAIR",data=df)
plt.xticks(rotation = 90)
plt.legend(loc=1)

plt.xticks(rotation = 90)
sns.barplot(x="ALIGN",y="APPEARANCES",data=df)
plt.xticks(rotation = 90)


df1=df.groupby(by=["ALIGN"]).sum()
labels=[]
for i in df1.index:
    labels.append(i)
colors=sns.color_palette("bright")
plt.pie(df1["APPEARANCES"],labels=labels,autopct="%0.0f%%")
plt.show()

df.corr()
plt.subplots(figsize=(12,7))
sns.heatmap(df.corr(),annot=True)

sns.histplot(data=df,x ='ALIVE',hue='ALIGN')

sns.kdeplot(x="Year", data = df,hue='ALIVE')
```
## Output:

![image](https://user-images.githubusercontent.com/93482485/204822541-485d2d25-0d6b-4938-9be5-a260f58586fe.png)
![image](https://user-images.githubusercontent.com/93482485/204822602-e9a94c21-4355-43f7-b172-37d5e9bac72f.png)
![image](https://user-images.githubusercontent.com/93482485/204822691-f399ce3f-02b5-45b3-922d-5ee8f739deba.png)
![image](https://user-images.githubusercontent.com/93482485/204822722-8087e1ae-bb61-454f-9358-82ada26a7619.png)
![image](https://user-images.githubusercontent.com/93482485/204822776-585f5f9f-c8d6-47d0-91c7-5b7a57dd097e.png)
![image](https://user-images.githubusercontent.com/93482485/204822857-475f7d6a-1826-4cf3-9857-051b1f43ece2.png)
![image](https://user-images.githubusercontent.com/93482485/204822814-b55fea3c-9acb-4304-b7ec-8d547432b611.png)
![image](https://user-images.githubusercontent.com/93482485/204822904-f8fac28e-2418-435d-a02f-8247a142f9f4.png)
![image](https://user-images.githubusercontent.com/93482485/204822931-1ff15c0f-1714-497b-9329-13705c9cdeb0.png)
![image](https://user-images.githubusercontent.com/93482485/204822955-206ef91c-7081-472d-8f90-46fbadb3df3d.png)
![image](https://user-images.githubusercontent.com/93482485/204822993-3e999985-7353-416d-8d5d-581fc3ef2b12.png)
![image](https://user-images.githubusercontent.com/93482485/204823042-b0112f9b-ca6f-42b7-89ba-283f41b817a3.png)
![image](https://user-images.githubusercontent.com/93482485/204823094-ef775392-b546-49e5-8498-57ae8a6adb30.png)
![image](https://user-images.githubusercontent.com/93482485/204823119-0dbe5f6e-f4a5-4b6e-a1ae-3bd70207f820.png)
![image](https://user-images.githubusercontent.com/93482485/204823149-be3e4a1f-4306-470f-b5c4-e1228e929c2b.png)
![image](https://user-images.githubusercontent.com/93482485/204823169-34937a2f-905b-4b06-9286-7c223817aa8c.png)
![image](https://user-images.githubusercontent.com/93482485/204823220-fac14658-982d-4439-9f75-d1fa962c7c88.png)
![image](https://user-images.githubusercontent.com/93482485/204823258-2eebaa17-f1ed-4a3e-a343-29c8f45a3573.png)
![image](https://user-images.githubusercontent.com/93482485/204823283-602c52b7-ce4f-4e66-bd76-0e7e01b94386.png)
![image](https://user-images.githubusercontent.com/93482485/204823311-32c5a92c-7edf-459e-a079-c376dde7c585.png)
![image](https://user-images.githubusercontent.com/93482485/204823348-6976d571-d2c4-4400-a9fa-360433568f59.png)
![image](https://user-images.githubusercontent.com/93482485/204823379-8c1180cc-a601-443c-93fc-67f691252fa6.png)
![image](https://user-images.githubusercontent.com/93482485/204823421-fd5065f3-3760-4e1d-8259-b6ce41551346.png)
![image](https://user-images.githubusercontent.com/93482485/204823459-c5358b08-9f0f-4b5e-a998-b2ade85e029f.png)


## Conclusion:
Data science is employed in many spheres of human life. 
The value of the algorithms and their efficiency can hardly be underestimated.
The use of data science in the field of media and entertainment has become an art.
It is no longer enough just to spread news, rumors or offer entertaining activities.
A company should reach the interaction with a customer, evoke feelings and emotions and make a desirable impact. 
The ability of data science to collect, process, analyze, store, provide recommendations is a huge benefit for the media and the entertainers.

