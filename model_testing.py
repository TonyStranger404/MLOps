import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def main():
    # Загрузка тестовых данных
    X_test = pd.read_csv('F:/Games/JetBrains/Projects/MLOps/MLOps/test/X_val_scaled.csv')
    y_test = pd.read_csv('F:/Games/JetBrains/Projects/MLOps/MLOps/test/y_val.csv')

    # Загрузка обученной модели
    model = joblib.load('F:/Games/JetBrains/Projects/MLOps/MLOps/model/linear_regression_model.pkl')

    # Предсказание тестовых данных
    y_pred = model.predict(X_test)

    # Вычисление метрик
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')


if __name__ == '__main__':
    main()