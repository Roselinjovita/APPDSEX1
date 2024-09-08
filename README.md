# APPDSEX1
Implementing Data Preprocessing and Data Analysis

## AIM:
To implement Data analysis and data preprocessing using a data set

## ALGORITHM:
Step 1: Import the data set necessary

Step 2: Perform Data Cleaning process by analyzing sum of Null values in each column a dataset.

Step 3: Perform Categorical data analysis.

Step 4: Use Sklearn tool from python to perform data preprocessing such as encoding and scaling.

Step 5: Implement Quantile transfomer to make the column value more normalized.

Step 6: Analyzing the dataset using visualizing tools form matplot library or seaborn.

## CODING AND OUTPUT:

#### NAME : ROSELIN MARY JOVITA.S
#### REG NO : 212222230122

### DATA CLEANING PROCESS

```
import pandas as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("/content/Toyota.csv")

df.head()
df.tail()
df.info()
df.describe()
df.shape()
df.fillna(method="ffill",inplace=True)
df.isnull().sum()

```
<table>
<tr>
<td>
      
 ![Screenshot 2024-09-08 110009](https://github.com/user-attachments/assets/b381c91c-08f9-4747-8bdc-b5bd2bbe5b16)
      
</td>
<td>
  
![Screenshot 2024-09-08 110023](https://github.com/user-attachments/assets/e62d310b-a221-4a67-8d06-344359a0aa47)
      
</td>
</tr>
</table>

<table>
<tr>
<td>
  
![Screenshot 2024-09-08 110034](https://github.com/user-attachments/assets/ece99856-beb3-45a5-bda9-abf0baa1a24d)

</td>
<td>
  
![Screenshot 2024-09-08 113731](https://github.com/user-attachments/assets/ac9f9965-b134-49b8-acb4-d0d5983590d1)

</tr>
</table>

![Screenshot 2024-09-08 110050](https://github.com/user-attachments/assets/aad4526b-9734-44dc-a8bc-940adaaf828a)


BEFORE & AFTER FILLING THE NULL VALUES<br> 
<table>
<tr>
<td>
  
![Screenshot 2024-09-08 110108](https://github.com/user-attachments/assets/3496ea34-a8f4-4533-83f4-7be93f84a3a6)
</td>
<td>
  
![Screenshot 2024-09-08 110813](https://github.com/user-attachments/assets/6bbc9e65-b9c7-48a4-b522-ff2fdfce172f)
</td>
</tr>
</table>

### OUTLIER DETECTION & REMOVAL 
```
sns.boxplot(df["Age"])
q1, q3 = df['Age'].quantile([0.25,0.75])
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
df = df[(df['Age'] >= lower_limit) & (df['Age'] <= upper_limit)]
sns.boxplot(df["Age"])

sns.boxplot(df["Price"])
q1, q3 = df['Price'].quantile([0.25,0.75])
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
df = df[(df['Price'] >= lower_limit) & (df['Price'] <= upper_limit)]
sns.boxplot(df["Price"])
```
<table>
<tr>
<td>
  BEFORE REMOVING OUTLIERS
  
![Screenshot 2024-09-08 111133](https://github.com/user-attachments/assets/36fd3c8b-035e-4592-b072-5d19361b02d3)

![Screenshot 2024-09-08 113206](https://github.com/user-attachments/assets/8357787e-f7b5-47bc-b743-cc6547a67b30)


</td>
<td>
  AFTER REMOVING OUTLIERS
  
![Screenshot 2024-09-08 111144](https://github.com/user-attachments/assets/9a5c2714-efb2-4e73-9767-5ce3955ebc15)

![Screenshot 2024-09-08 113214](https://github.com/user-attachments/assets/e6c95a7d-82de-426c-85af-cf0c8356c781)


</td>
</tr>   
</table>

### CATEGORICAL ANALYSIS

```
fuel_type_counts = df['FuelType'].value_counts()
print(fuel_type_counts)
fuel_type_counts.plot(kind='bar',color='pink')
plt.title('Count of Fuel Types')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='x', linestyle='solid', alpha=0.7)
plt.show()
```
![Screenshot 2024-09-08 115000](https://github.com/user-attachments/assets/e47b5fe2-e1a7-478b-aab4-f5ce46b17460)

![Screenshot 2024-09-08 115008](https://github.com/user-attachments/assets/c9450fb1-332a-45ff-95cc-a91444b9fd50)


### BIVARIATE AND MULTIVARIATE ANALYSIS

```
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Price', hue='FuelType', data=df)
plt.title('Bivariate Analysis: Age vs. Price by Fuel Type')
plt.xlabel('Age')
plt.ylabel('Price')
plt.show()

sns.pairplot(df[['Age', 'Price', 'KM', 'FuelType']], hue='FuelType')
plt.show()
```
<table>
<tr>
<td>
BIVARIATE ANALYSIS
  
![Screenshot 2024-09-08 115155](https://github.com/user-attachments/assets/95c0349d-2625-4450-9f65-f6ed0f034bcc)

</td>
<td>
MULTIVARIATE ANALYSIS

![Screenshot 2024-09-08 115203](https://github.com/user-attachments/assets/f0b98fa6-a3d9-4091-a945-d6fc8e32ef88)

      
</td>
</tr>
</table>

### ENCODING & FEATURE TRANSFORMATION

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['FuelType']=le.fit_transform(df['FuelType'])
df['FuelType']
df.tail()
df=pd.read_csv("Toyota.csv",index_col=0,na_values=["??","????"])
df["MetColor"]=df["MetColor"].astype('object')
df["Automatic"]=df['Automatic'].astype('object')
print(np.unique(df['Doors']))
df['Doors'].replace("three",3,inplace=True)
df['Doors'].replace("four",4,inplace=True)
df['Doors'].replace("five",5,inplace=True)
df['Doors']

df.skew()
np.log(df["KM"] )
```
LABEL ENCODER
<table>
<tr>
<td>
  
![Screenshot 2024-09-08 121834](https://github.com/user-attachments/assets/a3f61846-6e44-4128-8ba6-5a7d04cf617b)

</td>
<td>
  
![Screenshot 2024-09-08 121850](https://github.com/user-attachments/assets/0d2dd3a4-0a55-4a45-ab41-1adfd736cf1c)

</td>
</tr>
</table

FEATURE TRANSFORMATION

<table>
<tr>
<td>
  
 ![Screenshot 2024-09-08 121902](https://github.com/user-attachments/assets/2c07338e-72d1-4ae5-b1fa-a3f673a9b30f)
 
</td>
<td>
  
![Screenshot 2024-09-08 121918](https://github.com/user-attachments/assets/b6660399-c20e-4ab0-870e-9b100040bd73)

</td>
    
</tr>
</table>

BEFORE :

![Screenshot 2024-09-08 131806](https://github.com/user-attachments/assets/3ad34dad-3931-4347-adca-c2a2c72f30b0)

AFTER :

![Screenshot 2024-09-08 131755](https://github.com/user-attachments/assets/e28fa9f2-f08e-4f6a-9af0-9c423fcbf99a)



### DATA VISUALIZATION

```
sns.violinplot(x='Age', y='FuelType', data=df)
plt.show()

dfc['FuelType'] = dfc['FuelType'].astype('category')
dfc['FuelType'] = dfc['FuelType'].cat.codes
data = dfc.drop(columns=['Automatic'])
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```
<table>
<tr>
<td>

VIOLIN PLOT

![Screenshot 2024-09-08 132106](https://github.com/user-attachments/assets/bb7407c1-ee69-4464-a336-81abe1f56799)

</td>
<td>

HEATMAP

![Screenshot 2024-09-08 132115](https://github.com/user-attachments/assets/6f4209de-136b-40c1-877c-a2be55885ab7)

</td>
</tr>
</table>








## RESULT:
Thus Data analysis and Data preprocessing implemeted using a dataset.
