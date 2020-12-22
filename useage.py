import pandas as pd
from g_naive_bayes import gnb

data = pd.DataFrame({
    'gender': ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female'],
    'height': [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75],
    'weight': [180, 190, 170, 165, 100, 150, 130, 150],
    'foot_size': [12, 11, 12, 10, 6, 8, 7, 9]
})

x = data[['height', 'weight', 'foot_size']]
y = data['gender']

nb = gnb()
nb.fit(x, y)

x_test = pd.Series({'height': 6, 'weight': 130, 'foot_size': 8})

print(nb.predict(x_test))
