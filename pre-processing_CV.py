import pandas as pd

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

train = train.drop(["Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned", "Awards/honours", "Avg quality score"], axis = 1)
test = test.drop(["Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned", "Awards/honours", "Avg quality score"], axis = 1)

for i in train.columns:
    if i== "File name":
        continue
    for j in range(0, len(train[i])):
        train.loc[j, i] = float(train[i][j])
        # print(type(train[i][j]))

for i in train.columns:
    # train[i] = float(train[i])
    if i == "File name":
        continue
    train[i] = (train[i] - train[i].min()) / (train[i].max() - train[i].min())

for i in test.columns:
    if i== "File name":
        continue
    for j in range(0, len(test[i])):
        test.loc[j, i] = float(test[i][j])
        # print(type(train[i][j]))


for i in test.columns:
    if i == "File name":
        continue
    test[i] = (test[i] - test[i].min()) / (test[i].max() - test[i].min())




