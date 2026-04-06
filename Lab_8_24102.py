import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy.linalg import pinv


# -----------------------------
# A1: BASIC UNITS
# -----------------------------
def summation_unit(x, w, b):
    return np.dot(x, w) + b


def step(x):
    return 1 if x >= 0 else 0


def bipolar_step(x):
    return 1 if x >= 0 else -1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return max(0, x)


def leaky_relu(x):
    return x if x > 0 else 0.01 * x


def compute_error(y, y_pred):
    return y - y_pred


# -----------------------------
# PERCEPTRON TRAINING
# -----------------------------
def train_perceptron(X, y, w, b, lr, activation, max_epochs=1000, threshold=0.002):
    errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X)):
            net = summation_unit(X[i], w, b)
            output = activation(net)
            err = compute_error(y[i], output)

            w = w + lr * err * X[i]
            b = b + lr * err

            total_error += err ** 2

        errors.append(total_error)

        if total_error <= threshold:
            break

    return w, b, errors, epoch + 1


# -----------------------------
# A3: ACTIVATION COMPARISON
# -----------------------------
def compare_activations(X, y, w, b, lr):
    activations = {
        "Step": step,
        "Bipolar": bipolar_step,
        "Sigmoid": lambda x: 1 if sigmoid(x) >= 0.5 else 0,
        "ReLU": lambda x: 1 if relu(x) > 0 else 0,
    }

    results = {}

    for name, fn in activations.items():
        _, _, _, epochs = train_perceptron(X, y, w.copy(), b, lr, fn)
        results[name] = epochs

    return results


# -----------------------------
# A4: LEARNING RATE ANALYSIS
# -----------------------------
def learning_rate_experiment(X, y, w, b):
    rates = np.arange(0.1, 1.1, 0.1)
    epochs_list = []

    for lr in rates:
        _, _, _, epochs = train_perceptron(X, y, w.copy(), b, lr, step)
        epochs_list.append(epochs)

    return rates, epochs_list


# -----------------------------
# A6: CUSTOMER DATASET
# -----------------------------
def customer_dataset():
    X = np.array([
        [20, 6, 2, 386], [16, 3, 6, 289], [27, 6, 2, 393], [19, 1, 2, 110],
        [24, 4, 2, 280], [22, 1, 5, 167], [15, 4, 2, 271], [18, 4, 2, 274],
        [21, 1, 4, 148], [16, 2, 4, 198]
    ])
    y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    return X, y


# -----------------------------
# A7: PSEUDO-INVERSE METHOD
# -----------------------------
def pseudo_inverse_solution(X, y):
    X_aug = np.c_[np.ones(len(X)), X]
    weights = pinv(X_aug).dot(y)
    return weights


# -----------------------------
# A8: BACKPROPAGATION NETWORK
# -----------------------------
def train_backprop(X, y, lr=0.05, max_epochs=1000, threshold=0.002):
    np.random.seed(0)

    W1 = np.random.rand(2, 2)
    W2 = np.random.rand(2, 1)

    errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X)):
            x = X[i].reshape(1, -1)
            target = y[i]

            hidden = sigmoid(np.dot(x, W1))
            output = sigmoid(np.dot(hidden, W2))

            err = target - output
            total_error += float(err ** 2)

            d_output = err * output * (1 - output)
            d_hidden = d_output.dot(W2.T) * hidden * (1 - hidden)

            W2 += lr * hidden.T.dot(d_output)
            W1 += lr * x.T.dot(d_hidden)

        errors.append(float(total_error))

        if total_error <= threshold:
            break

    return W1, W2, errors


# -----------------------------
# OPTIONAL O1
# -----------------------------
def optional_learning_rate_analysis(X, y, w, b):
    rates = np.arange(0.1, 1.1, 0.1)
    results = {"Sigmoid": [], "ReLU": []}

    for lr in rates:
        _, _, _, ep_sig = train_perceptron(
            X, y, w.copy(), b, lr,
            lambda x: 1 if sigmoid(x) >= 0.5 else 0
        )

        _, _, _, ep_relu = train_perceptron(
            X, y, w.copy(), b, lr,
            lambda x: 1 if relu(x) > 0 else 0
        )

        results["Sigmoid"].append(ep_sig)
        results["ReLU"].append(ep_relu)

    return rates, results


# -----------------------------
# OPTIONAL O2
# -----------------------------
def optional_activation_comparison(X, y, w, b, lr):
    activations = {
        "Sigmoid": lambda x: 1 if sigmoid(x) >= 0.5 else 0,
        "ReLU": lambda x: 1 if relu(x) > 0 else 0,
        "Tanh": lambda x: 1 if tanh(x) >= 0 else 0,
        "LeakyReLU": lambda x: 1 if leaky_relu(x) > 0 else 0
    }

    results = {}

    for name, fn in activations.items():
        _, _, _, epochs = train_perceptron(X, y, w.copy(), b, lr, fn)
        results[name] = epochs

    return results


# -----------------------------
# OPTIONAL O3
# -----------------------------
def optional_backprop_experiment(X, y):
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    results = {}

    for lr in learning_rates:
        _, _, errors = train_backprop(X, y, lr)
        results[lr] = len(errors)

    return results


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():

    # AND Gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    w = np.array([0.2, -0.75])
    b = 10
    lr = 0.05

    # A2
    _, _, err, epochs = train_perceptron(X, y, w.copy(), b, lr, step)
    print("A2 Epochs:", epochs)
    plt.plot(err)
    plt.title("A2 Error vs Epochs")
    plt.show()

    # A3
    print("A3 Activation Comparison:", compare_activations(X, y, w.copy(), b, lr))

    # A4
    rates, ep = learning_rate_experiment(X, y, w.copy(), b)
    plt.plot(rates, ep)
    plt.title("A4 Learning Rate vs Epochs")
    plt.show()

    # A5 XOR
    y_xor = np.array([0, 1, 1, 0])
    _, _, _, xor_epochs = train_perceptron(X, y_xor, w.copy(), b, lr, step)
    print("A5 XOR Epochs:", xor_epochs)

    # A6 Customer Data
    Xc, yc = customer_dataset()
    w_c = np.random.rand(Xc.shape[1])

    _, _, _, cust_epochs = train_perceptron(
        Xc, yc, w_c, 0, 0.01,
        lambda x: 1 if sigmoid(x) >= 0.5 else 0
    )
    print("A6 Customer Epochs:", cust_epochs)

    # A7
    print("A7 Pseudo-inverse Weights:", pseudo_inverse_solution(Xc, yc))

    # A8 Backprop
    _, _, err_bp = train_backprop(X, y)
    plt.plot(err_bp)
    plt.title("A8 Backprop Error")
    plt.show()

    # A11
    clf = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    clf.fit(X, y)
    print("A11 AND Prediction:", clf.predict(X))

    clf.fit(X, y_xor)
    print("A11 XOR Prediction:", clf.predict(X))

    # Optional O1
    rates, results = optional_learning_rate_analysis(X, y, w.copy(), b)
    plt.plot(rates, results["Sigmoid"], label="Sigmoid")
    plt.plot(rates, results["ReLU"], label="ReLU")
    plt.legend()
    plt.title("O1 Learning Rate Analysis")
    plt.show()

    # Optional O2
    print("O2 Activation Comparison:", optional_activation_comparison(X, y, w.copy(), b, lr))

    # Optional O3
    print("O3 Backprop Learning Rate Analysis:", optional_backprop_experiment(X, y))


if __name__ == "__main__":
    main()