import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from sklearn.neural_network import MLPClassifier


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
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def compute_error(y, y_pred):
    return y - y_pred


# -----------------------------
# A2: PERCEPTRON
# -----------------------------
def train_perceptron(X, y, w, b, lr, activation, max_epochs=1000, threshold=0.002):
    errors = []

    for epoch in range(max_epochs):
        total_error = 0.0

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
# A3
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
# A4
# -----------------------------
def learning_rate_experiment(X, y, w, b):
    rates = np.arange(0.1, 1.1, 0.1)
    epochs_list = []

    for lr in rates:
        _, _, _, epochs = train_perceptron(X, y, w.copy(), b, lr, step)
        epochs_list.append(epochs)

    return rates, epochs_list


# -----------------------------
# A6
# -----------------------------
def customer_dataset():
    X = np.array([
        [20, 6, 2, 386], [16, 3, 6, 289], [27, 6, 2, 393], [19, 1, 2, 110],
        [24, 4, 2, 280], [22, 1, 5, 167], [15, 4, 2, 271], [18, 4, 2, 274],
        [21, 1, 4, 148], [16, 2, 4, 198]
    ], dtype=float)

    y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=float)

    return X, y


# -----------------------------
# A7
# -----------------------------
def pseudo_inverse_solution(X, y):
    X_aug = np.c_[np.ones(len(X)), X]
    return pinv(X_aug).dot(y)


# -----------------------------
# A8
# -----------------------------
def train_backprop(X, y, lr=0.05, max_epochs=1000, threshold=0.002):
    np.random.seed(0)

    W1 = np.random.rand(X.shape[1], 2)
    W2 = np.random.rand(2, 1)

    errors = []

    for epoch in range(max_epochs):
        total_error = 0.0

        for i in range(len(X)):
            x = X[i].reshape(1, -1)
            target = y[i]

            hidden = sigmoid(np.dot(x, W1))
            output = sigmoid(np.dot(hidden, W2))

            err = target - output
            total_error += (err ** 2).item()

            d_output = err * output * (1 - output)
            d_hidden = d_output.dot(W2.T) * hidden * (1 - hidden)

            W2 += lr * hidden.T.dot(d_output)
            W1 += lr * x.T.dot(d_hidden)

        errors.append(total_error)

        if total_error <= threshold:
            break

    return W1, W2, errors


# -----------------------------
# A9
# -----------------------------
def run_A9_XOR(X, y_xor, w, b, lr):
    _, _, errors, epochs = train_perceptron(X, y_xor, w.copy(), b, lr, step)
    return errors, epochs


# -----------------------------
# A10
# -----------------------------
def train_perceptron_two_outputs(X, y, lr, max_epochs=1000, threshold=0.002):
    y_encoded = np.array([[1, 0] if val == 0 else [0, 1] for val in y], dtype=float)

    w = np.random.rand(X.shape[1], 2)
    b = np.zeros(2)

    errors = []

    for epoch in range(max_epochs):
        total_error = 0.0

        for i in range(len(X)):
            net = np.dot(X[i], w) + b
            output = np.array([step(net[0]), step(net[1])])

            err = y_encoded[i] - output

            w += lr * np.outer(X[i], err)
            b += lr * err

            total_error += np.sum(err ** 2)

        errors.append(total_error)

        if total_error <= threshold:
            break

    return w, b, errors, epoch + 1


# -----------------------------
# O1
# -----------------------------
def optional_O1(X, y, w, b):
    rates = np.arange(0.1, 1.1, 0.1)
    sig, relu_vals = [], []

    for lr in rates:
        _, _, _, ep1 = train_perceptron(
            X, y, w.copy(), b, lr,
            lambda x: 1 if sigmoid(x) >= 0.5 else 0
        )
        _, _, _, ep2 = train_perceptron(
            X, y, w.copy(), b, lr,
            lambda x: 1 if relu(x) > 0 else 0
        )

        sig.append(ep1)
        relu_vals.append(ep2)

    return rates, sig, relu_vals


# -----------------------------
# O2
# -----------------------------
def optional_O2(X, y, w, b, lr):
    activations = {
        "Sigmoid": lambda x: 1 if sigmoid(x) >= 0.5 else 0,
        "ReLU": lambda x: 1 if relu(x) > 0 else 0,
        "Tanh": lambda x: 1 if tanh(x) >= 0 else 0,
        "LeakyReLU": lambda x: 1 if leaky_relu(x) > 0 else 0
    }

    results = {}
    for name, fn in activations.items():
        _, _, _, ep = train_perceptron(X, y, w.copy(), b, lr, fn)
        results[name] = ep

    return results


# -----------------------------
# O3
# -----------------------------
def optional_O3(X, y):
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    results = {}

    for lr in learning_rates:
        _, _, errors = train_backprop(X, y, lr)
        results[lr] = len(errors)

    return results


# -----------------------------
# MAIN
# -----------------------------
def main():

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_and = np.array([0, 0, 0, 1], dtype=float)
    y_xor = np.array([0, 1, 1, 0], dtype=float)

    w = np.array([0.2, -0.75], dtype=float)
    b = 10.0
    lr = 0.05

    # A2
    _, _, errors, epochs = train_perceptron(X, y_and, w.copy(), b, lr, step)
    print("A2 Epochs:", epochs)

    plt.figure(figsize=(6,4))
    plt.plot(errors)
    plt.title("A2: Error vs Epochs (AND Gate - Step Activation)")
    plt.xlabel("Epochs")
    plt.ylabel("Sum Squared Error")
    plt.grid(True)
    plt.show()

    # A3
    print("A3:", compare_activations(X, y_and, w.copy(), b, lr))

    # A4
    rates, ep = learning_rate_experiment(X, y_and, w.copy(), b)

    plt.figure(figsize=(6,4))
    plt.plot(rates, ep, marker='o')
    plt.title("A4: Learning Rate vs Epochs (AND Gate)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs to Converge")
    plt.grid(True)
    plt.show()

    # A5
    _, _, _, xor_epochs = train_perceptron(X, y_xor, w.copy(), b, lr, step)
    print("A5 XOR Epochs:", xor_epochs)

    # A6
    Xc, yc = customer_dataset()
    _, _, _, cust_epochs = train_perceptron(
        Xc, yc, np.random.rand(Xc.shape[1]), 0.0, 0.01,
        lambda x: 1 if sigmoid(x) >= 0.5 else 0
    )
    print("A6:", cust_epochs)

    # A7
    print("A7:", pseudo_inverse_solution(Xc, yc))

    # A8
    _, _, err_bp = train_backprop(X, y_and)

    plt.figure(figsize=(6,4))
    plt.plot(err_bp)
    plt.title("A8: Backpropagation Error vs Epochs (AND Gate)")
    plt.xlabel("Epochs")
    plt.ylabel("Sum Squared Error")
    plt.grid(True)
    plt.show()

    # A9
    _, ep9 = run_A9_XOR(X, y_xor, w.copy(), b, lr)
    print("A9:", ep9)

    # A10
    _, _, err10, ep10 = train_perceptron_two_outputs(X, y_and, lr)
    print("A10:", ep10)

    plt.figure(figsize=(6,4))
    plt.plot(err10)
    plt.title("A10: Error vs Epochs (Two Output Perceptron)")
    plt.xlabel("Epochs")
    plt.ylabel("Sum Squared Error")
    plt.grid(True)
    plt.show()

    # A11
    clf = MLPClassifier(hidden_layer_sizes=(2,), max_iter=2000)

    clf.fit(X, y_and)
    print("A11 AND:", clf.predict(X))

    clf.fit(X, y_xor)
    print("A11 XOR:", clf.predict(X))

    # O1
    r, s, r2 = optional_O1(X, y_and, w.copy(), b)

    plt.figure(figsize=(6,4))
    plt.plot(r, s, marker='o', label="Sigmoid Activation")
    plt.plot(r, r2, marker='s', label="ReLU Activation")
    plt.title("O1: Learning Rate vs Epochs for Different Activations")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs to Converge")
    plt.legend()
    plt.grid(True)
    plt.show()

    # O2
    print("O2:", optional_O2(X, y_and, w.copy(), b, lr))

    # O3
    print("O3:", optional_O3(X, y_and))


if __name__ == "__main__":
    main()
