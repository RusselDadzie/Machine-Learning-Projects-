import os                                 #import libraries (ensure all necessary build in and runtime dependencies are available) 
import numpy as np
import plotly.graph_objects as go  
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics 
import scipy.stats
import math 

path = r"C:\Users\USER\.cache\kagglehub\datasets\timoboz\tesla-stock-data-from-2010-to-2020\versions\1"


file_name = "tesla.csv"                    #open and read the data file 
full_path = f"{path}/{file_name}"
df = pd.read_csv(full_path)
print(df.head()) 
print(df.shape)                     

df = df.drop (columns=[                  
    'Adj Close'
],axis=1)
print (df.head())                          #remove columns you wont need 

print(df.duplicated().sum().any())         #clean the data by checking for duplicates or null values 
print(df.isnull().values.any()) 

num_df = df.select_dtypes(include = [np.number]) 
print(num_df.corr)

plt.figure(figsize=(16,8))                 #visualize data 
sns.heatmap(num_df.corr(), cmap='Blues', annot = True)
plt.title('Correlation')


sns.pairplot(df)


f, axes = plt.subplots(1,4)                 #use a boxplot to check how many extremeties are in the dataset 
sns.boxplot (y='Open', data=df, ax=axes[0])
sns.boxplot (y='High', data=df, ax=axes[1])
sns.boxplot (y='Low', data=df, ax=axes[2])
sns.boxplot (y='Close', data=df, ax=axes[3])


figure = go.Figure(data = [go.Candlestick(x=df["Date"],
                                          open=df["Open"],high=df["High"],
                                          low=df["Low"], close=df["Close"])])
figure.update_layout(title = "Google Stock Price Analysis", xaxis_rangeslider_visible=False)
figure.show()


X = df[['Open','High','Low','Volume']] #splitting dataset for our input variables and output (independent variables and dependent)
Y = df[['Close']] 

from sklearn.model_selection import train_test_split                                           #split test and train data (20%,80%)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
print('Train:', X_train.shape)
print('Test', X_test.shape) 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,accuracy_score 
import statsmodels.api as sm 

#train the model
regressor = LinearRegression()

model = regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)

print('Predicted value shape:',y_pred.shape)

#validate our line of best fit 
print('Model Coefficients:', regressor.coef_)
print('Model Intercept', regressor.intercept_)

Y_test_flat = Y_test.to_numpy().flatten()  # Convert to NumPy and flatten
y_pred_flat = y_pred.flatten()

# Create DataFrame to compare Actual vs Predicted
dfr = pd.DataFrame({
    'Actual Price': Y_test_flat,
    'Predicted Price': y_pred_flat
})

print(dfr) 

print(dfr.describe())

residual = Y_test - y_pred                     #test model validity with visualization of distribution plot to check the skewness 
sns.displot(residual)
plt.show() 

p_value = scipy.stats.norm.sf(abs(1.67))
print('p_value is :', p_value)


regression_confidence = regressor.score(X_test,Y_test)      #checking the regression score
print(regression_confidence) 

print(metrics.mean_absolute_error(Y_test,y_pred))
print(metrics.mean_squared_error(Y_test,y_pred))
print (math.sqrt(metrics.mean_squared_error(Y_test,y_pred)))


#check for model accuracy 
A1 = abs(y_pred-Y_test)
A2 = 100 * ( A1 /Y_test) 
accuracy = 100 - np.mean(A2)
print('Model Accuracy is :',round(accuracy,2),'%')

plt.scatter(Y_test,y_pred, color = 'Darkblue') 
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show() 

graph = dfr.head(200)
graph.plot(kind='bar') 
plt.show()

#create function to allow you to place random inputs for predicted close values
def test_model_with_inputs(model, scaler=None):
    print("\nTest the model with new independent variables")
    # Prompt for user input or generate random values within reasonable ranges
    try:
        open_price = float(input("Enter the Open price (or press Enter for a random value): ") or np.random.uniform(18, 22))
        high_price = float(input("Enter the High price (or press Enter for a random value): ") or np.random.uniform(25, 35))
        low_price = float(input("Enter the Low price (or press Enter for a random value): ") or np.random.uniform(14, 19))
        volume = float(input("Enter the Volume (or press Enter for a random value): ") or np.random.uniform(15000000, 20000000))
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return
    
# Combine input variables into a DataFrame
    input_data = np.array([[open_price, high_price, low_price, volume]])
    predicted_close_price = model.predict(input_data)
    
    print("\nInputs provided:")
    print(f"Open Price: {open_price}")
    print(f"High Price: {high_price}")
    print(f"Low Price: {low_price}")
    print(f"Volume: {volume}")
    print("\nPredicted Close Price:")
    print(predicted_close_price[0])

# Test the function with your trained model
test_model_with_inputs(regressor)
