# Diamond Price Prediction using Machine Learning

This repository demonstrates the process of predicting the price of diamonds based on various attributes using a **Random Forest Regressor** model. The dataset utilized contains information about different diamonds, including features such as carat size, cut quality, depth, and dimensions (length, width, and height). The workflow includes exploratory data analysis, data preprocessing, feature engineering, and model training, followed by making predictions based on user input.

## Overview of the Code

The code involves the following major steps:

### 1. **Data Loading and Initial Exploration**
The dataset is loaded from a CSV file (`diamonds.csv`) using the `pandas` library:

```python
data = pd.read_csv("diamonds.csv")
print(data.head())
```

This step prints the first five rows of the dataset, allowing us to inspect the available columns. A column named `Unnamed: 0` is dropped as it serves no useful purpose in the analysis:

```python
data = data.drop("Unnamed: 0", axis=1)
```

### 2. **Data Visualization**
A scatter plot is generated to visualize the relationship between the `carat` and `price` of diamonds, with the `depth` representing the size of the markers. The color represents the `cut` quality of the diamonds. A trendline is also fitted to the data:

```python
figure = px.scatter(data_frame=data, x="carat", y="price", size="depth", color="cut", trendline="ols")
figure.show()
```

### 3. **Feature Engineering: Calculating Diamond Size**
A new feature called `size` is created by multiplying the dimensions of the diamond (i.e., length, width, and depth):

```python
data["size"] = data["x"] * data["y"] * data["z"]
```

Any missing values in the `size` column are filled with 0 to prevent errors during model training:

```python
data['size'] = data['size'].fillna(0)
```

### 4. **Additional Visualizations**
Box plots are generated to explore the distribution of diamond prices based on their `cut` quality, with the additional grouping by `color` and `clarity`:

```python
fig = px.box(data, x="cut", y="price", color="color")
fig.show()

fig = px.box(data, x="cut", y="price", color="clarity")
fig.show()
```

### 5. **Correlation Analysis**
A correlation matrix is computed to evaluate the relationships between numerical features in the dataset and the `price` of the diamonds:

```python
correlation = data.select_dtypes(include=np.number).corr()
print(correlation["price"].sort_values(ascending=False))
```

### 6. **Data Preprocessing: Handling Categorical Data**
The categorical feature `cut` is mapped to numerical values to facilitate model training:

```python
data["cut"] = data["cut"].map({"Ideal": 1, "Premium": 2, "Good": 3, "Very Good": 4, "Fair": 5})
```

### 7. **Model Training and Evaluation**
The dataset is split into features (`x`) and target (`y`), with the target being the `price` of the diamond:

```python
x = np.array(data[["carat", "cut", "size"]])
y = np.array(data[["price"]])
```

The data is then split into training and testing sets using an 80-20 split. Missing values in the target variable (`y`) are imputed using the mean strategy:

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
y = imputer.fit_transform(y)
```

A **Random Forest Regressor** model is trained using the training data:

```python
model = RandomForestRegressor()
model.fit(xtrain, ytrain)
```

### 8. **Predicting Diamond Price**
The user is prompted to input the `carat`, `cut`, and `size` of a diamond. The trained model then predicts the price based on these inputs:

```python
print("Enter House Details to Predict Rent")
a = float(input("Carat Size: "))
b = int(input("Cut Type (Ideal: 1, Premium: 2, Good: 3, Very Good: 4, Fair: 5): "))
c = float(input("Size: "))
features = np.array([[a, b, c]])
print("Predicted Diamond's Price = ", model.predict(features))
```

The model's prediction will be printed based on the user’s input.

