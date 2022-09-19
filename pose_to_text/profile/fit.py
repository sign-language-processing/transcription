import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def features(src_len, trg_len):
    return [src_len, trg_len, src_len ** 2, trg_len ** 2, src_len * trg_len]


def features_text():
    return ["|S|", "|T|", "|S|^2", "|T|^2", "|S|*|T|"]


def parse_line(line: str):  # src_len=35, trg_len=8, memory=230691840
    src_len, trg_len, memory = line.split(", ")
    src_len = int(src_len.split("=")[1])
    trg_len = int(trg_len.split("=")[1])
    memory = int(memory.split("=")[1])
    return {"src_len": src_len, "trg_len": trg_len, "memory": memory}


with open("output.txt", "r", encoding="utf-8") as f:
    lines = f.read().strip().split("\n")
data = [parse_line(line) for line in lines]

X = np.array([features(d["src_len"], d["trg_len"]) for d in data])
y = np.array([d["memory"] for d in data])
print(X[0], y[0])

# Split the data into training/testing sets
X_train = X[:-20]
X_test = X[-20:]

# Split the targets into training/testing sets
y_train = y[:-20]
y_test = y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
formula = " + ".join([f"{c:.2f} * {f}" for c, f in zip(regr.coef_, features_text())])
print(f"Memory = {regr.intercept_} + {formula}")

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

print([y_pred[i]/y_test[i] for i in range(len(y_test))])

# Plot outputs
plt.scatter(X_test[:, 2], y_test, color="black")
plt.plot(X_test[:, 2], y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())
plt.savefig("output.png")
