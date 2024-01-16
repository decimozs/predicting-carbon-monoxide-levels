# imported libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

# glboal variables
filePath = "C:/Users/acer/Documents/Platform Technologies Grouping Case Study/AirQualityUCI.xlsx"
dataSet = pd.read_excel(filePath)
summary = dataSet.describe()
# load the dataset
def loadTheDataset():
    print(dataSet)
    
# visualize the dataset
def visualizeTheDataset():
    summary_transposed = summary.T

    # Plot bar plots for mean and standard deviation
    summary_transposed = summary.T
    plt.figure(figsize=(12, 6))
    sns.barplot(x=summary_transposed.index[1:],  # Exclude 'Date' from the x-axis
        y=summary_transposed['mean'].iloc[1:], color='blue', alpha=0.7, label='Mean')
    sns.barplot(x=summary_transposed.index[1:],  # Exclude 'Date' from the x-axis
        y=summary_transposed['std'].iloc[1:], color='orange', alpha=0.7, label='Std Dev')
    plt.title('Air Quality Visualization Data Set')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

def evaluate_model(model, X_val, y_val, X_test, y_test):
    # Make predictions on the validation set
    predictions_val = model.predict(X_val)

    # Make predictions on the testing set
    predictions_test = model.predict(X_test)

    # Line chart of actual vs. predicted values for both sets
    plt.figure(figsize=(12, 8))

    plt.plot(y_val, predictions_val, 'o-', color='red', label='Validation Set', alpha=0.8, linewidth=2, markersize=8)
    plt.plot(y_test, predictions_test, 'o-', color='blue', label='Test Set', alpha=0.8, linewidth=2, markersize=8)

    plt.title(f'Model Evaluation: {model.__class__.__name__}', fontsize=16)
    plt.xlabel('Actual Carbon Monoxide Concentrations', fontsize=14)
    plt.ylabel('Predicted Carbon Monoxide Concentrations', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Evaluate predictions on the validation set
    mse_val = mean_squared_error(y_val, predictions_val)
    r2_val = r2_score(y_val, predictions_val)
    print(f"\nModel Evaluation: {model.__class__.__name__} on Validation Set")
    print(f'Mean Squared Error: {mse_val:.2f}')
    print(f'R-squared: {r2_val:.2f}')

    # Evaluate predictions on the new data set
    mse_test = mean_squared_error(y_test, predictions_test)
    r2_test = r2_score(y_test, predictions_test)
    print(f"\nModel Evaluation: {model.__class__.__name__} on New Data Set")
    print(f'Mean Squared Error: {mse_test:.2f}')
    print(f'R-squared: {r2_test:.2f}')
    
# predict carbon monoxide levels
def predictCarbonMonoxideLevels():
    features = dataSet.drop(columns=['CO(GT)', 'Date', 'Time'])
    target = dataSet['CO(GT)']

    # Split the dataset into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Initialize the models or spot check algorithms
    models = [LinearRegression(),
              DecisionTreeRegressor(),
              RandomForestRegressor(),
              GradientBoostingRegressor(),
              SVR()]

    for model in models:
        # Train the model on the training set
        model.fit(X_train, y_train)

        evaluate_model(model, X_val, y_val, X_test, y_test)

        # Cross-validate the model 
        # make predictions using new dataset
        cross_val_scores = cross_val_score(model, features, target, cv=5, scoring='neg_mean_squared_error')
        mean_cross_val_mse = np.mean(np.abs(cross_val_scores))
        print(f'Mean Cross-Validated MSE for {model.__class__.__name__}: {mean_cross_val_mse}\n')

# main menu
def menu():
    while True:
        print("Air Quality Model")
        print("Menu")
        print("[1] Load the dataset")
        print("[2] Visualize the dataset")
        print("[3] Predict Carbon Monoxide Levels Using Different Models")
        print("[4] Exit")
        
        choice = int(input("Enter your choice: "))
        
        try:
            if choice == 1:
                loadTheDataset()
                menu()
        
            elif choice == 2:
                visualizeTheDataset()
                menu()
            
            elif choice == 3:
                predictCarbonMonoxideLevels()
                menu()
            
            elif choice == 4:
                print("Program Terminated")
                menu()
            
            else:
                print("Invalid input, please input 1 and 4.")
        except:
            print("Invalid input, please input a valid integer!")
    
if __name__ == "__main__":
    menu()