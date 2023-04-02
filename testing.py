import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([3.0, 5.5, 8.1, 9.8, 10.6, 13.0, 15.4, 17.2, 18.9, 20.5,
                        23.0, 25.5, 27.0, 29.5, 31.2, 32.8, 35.3, 37.0, 38.7, 40.2,
                        42.6, 44.2, 45.9, 48.3,]).reshape(-1, 1)
scores = np.array([14.6, 20.4, 30.1, 33.9, 34.6, 45.7, 43.8, 52.3, 62.6, 63.9,
                        62.1, 79.7, 77.8, 86.3, 87.2, 89.1, 86.6, 93.4, 95.2, 97.0,
                        97.9, 84.4, 70.0, 90.5]).reshape(-1, 1)

model = LinearRegression()
# use 20 percent of data to test model
time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size = 0.2)
model.fit(time_train, score_train)
print(model.score(time_test, score_test))