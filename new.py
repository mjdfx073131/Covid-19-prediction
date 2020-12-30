from pandas import read_csv
import numpy as np

col_list = ["country_id", "deaths",
            "cases", "cases_100k", "cases_14_100k"]
series = read_csv('phase2_training_data.csv', usecols=col_list,
                  header=0, parse_dates=True, squeeze=True)

X = series.values

CA = np.where(X[:, 0] == "CA")[0]

CA_deaths = X[(CA[0]):(CA[-1]+1), 2]
CA_del_zero = np.where(CA_deaths != 0)[0]
CA_deaths_new = CA_deaths[CA_del_zero]

new_X = np.zeros(shape=(len(CA_deaths_new) - 2, 3))

for i in range(len(CA_deaths_new) - 2):
    new_X[i] = np.concatenate((np.array(CA_deaths_new[i: i+2]), [1]))

new_Y = CA_deaths_new[2:]

ca_w = np.linalg.inv((new_X.T @ new_X))@new_X.T@new_Y  # calculating w


x_2 = CA_deaths_new[-2:]

for i in range(11):
    predict = (np.concatenate((x_2,[1])) @ ca_w.T)
    new_x_2 = np.append(x_2[-1:], predict)
    x_2 = new_x_2
    print(predict)
