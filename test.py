import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def train():
    # Load data
    data = pd.read_csv("data/flight_01_data_dummy.csv")

    # Prepare data
    # Pivot the data
    pivot_data = data.pivot_table(index=['id', 'timestamp', 'engine_id', 'failure'],
                                  columns='reading_type',
                                  values='value').reset_index()

    # Drop any unnecessary columns
    pivot_data.drop(columns=['id', 'timestamp', 'engine_id'], inplace=True)
    x = pivot_data[['voltage', 'rotor_speed_rpm', 'thrust_newton', 'temperature_celsius']]
    x = x.ffill()
    x = x.bfill()
    y = pivot_data["failure"]

    # Data preprocessing
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Evaluate model
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)

    # Predict failure for new data points
    # Assuming "new_data" contains input features for prediction
    # Preprocess "new_data" using the same scaler
    # Then, use the trained model to make predictions
    # predicted_failure = model.predict(new_data)


if __name__ == "__main__":
    train()