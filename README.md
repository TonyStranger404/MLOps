# MLops

## HW1
Этапы
-python-скрипт (data_creation.py),  создает различные наборы данных, описывающие некий процесс (например, изменение дневной температуры).  
-Часть наборов данных сохраняется в папке «train», другая часть — в папке «test».  
-python-скрипт (model_preprocessing.py),  выполняет предобработку данных, например с помощью sklearn.preprocessing.StandardScaler.  
-python-скрипт (model_preparation.py),  создает и обучает модель машинного обучения на построенных данных из папки «train».  
-python-скрипт (model_testing.py), проверяет модель машинного обучения на построенных данных из папки «test».  
-bash-скрипт (pipeline.sh), последовательно запускает все python-скрипты.
