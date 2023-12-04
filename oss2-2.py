import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def sort_dataset(dataset_df):
    return dataset_df.sort_values(by='year', ascending=True)

def split_dataset(dataset_df):
    dataset_df['salary'] *= 0.001
    train_df, test_df = train_test_split(dataset_df, test_size=0.2, shuffle=False)
    return train_df, test_df

def extract_numerical_cols(dataset_df):
    return dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]

def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, Y_train)
    return dt_model.predict(X_test)

def train_predict_random_forest(X_train, Y_train, X_test):
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, Y_train)
    return rf_model.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
    svm_model = SVR()
    svm_model.fit(X_train, Y_train)
    return svm_model.predict(X_test)

def calculate_RMSE(labels, predictions):
    rmse = mean_squared_error(labels, predictions, squared=False)
    return rmse


if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
