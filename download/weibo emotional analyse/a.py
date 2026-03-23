import pandas as pd
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(type(a))
data = pd.DataFrame(a)
print(data)
save_path = "dataset/predict_result.csv"
with open(save_path, 'w', encoding='utf-8')as f:
    data.to_csv(save_path, index=False)
