import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib


def main():
    # Загрузка данных
    X_train = pd.read_csv('F:/Games/JetBrains/Projects/MLOps/MLOps/train/X_train_scaled.csv')
    y_train = pd.read_csv('F:/Games/JetBrains/Projects/MLOps/MLOps/train/y_train.csv')

    # Создание и обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Сохранение обученной модели
    joblib.dump(model, 'F:/Games/JetBrains/Projects/MLOps/MLOps/model/linear_regression_model.pkl')

    # Опционально: Вычисление и вывод метрики модели
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2}')


if __name__ == '__main__':
    main()