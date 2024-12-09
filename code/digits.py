import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

train = []
test = []

for line in open("../data/zip.train").readlines():
    chunks = line.split()
    label, rest = int(float(chunks[0])), list(map(float, chunks[1:]))
    train.append({"label": label, **{f"f_{i}": rest[i] for i in range(len(rest))}})
train = pd.DataFrame(train)

for line in open("../data/zip.test").readlines():
    chunks = line.split()
    label, rest = int(float(chunks[0])), list(map(float, chunks[1:]))
    test.append({"label": label, **{f"f_{i}": rest[i] for i in range(len(rest))}})
test = pd.DataFrame(test)


restrict = [2, 3]
train = train[train["label"].isin(restrict)]
test = test[test["label"].isin(restrict)]

train_X, train_y = (
    train[[col for col in train.columns if col != "label"]],
    train["label"],
)
test_X, test_y = test[[col for col in test.columns if col != "label"]], test["label"]

lm = LogisticRegression().fit(train_X, train_y)
print(lm.score(test_X, test_y))

nn = KNeighborsClassifier(n_neighbors=7).fit(train_X, train_y)
print(nn.score(test_X, test_y))
