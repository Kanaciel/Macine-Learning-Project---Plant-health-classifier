import matplotlib.pyplot as plt

train_loss = [
    0.5847, 0.2546, 0.2098, 0.1811, 0.1669,
    0.1509, 0.1314, 0.1118, 0.1154, 0.1066,
    0.0930, 0.0829, 0.0733, 0.0708, 0.0637
]

test_loss = [
    0.2342, 0.1588, 0.1460, 0.1287, 0.1273,
    0.1201, 0.1229, 0.1019, 0.0964, 0.0991,
    0.0952, 0.0864, 0.0876, 0.0797, 0.0854
]

train_acc = [
    82.43, 91.82, 93.00, 93.94, 94.33,
    94.69, 95.64, 96.01, 96.15, 96.30,
    96.85, 97.26, 97.60, 97.65, 97.79
]

test_acc = [
    92.90, 95.21, 95.13, 95.79, 95.98,
    95.98, 96.09, 96.59, 96.59, 96.86,
    96.89, 97.05, 97.19, 97.58, 97.22
]




epochs = range(1, len(train_loss) + 1)

plt.figure()
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, test_loss, label="Test Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Test Loss")
plt.legend()
plt.grid(True)


epochs = range(1, len(train_acc) + 1)

plt.figure()
plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, test_acc, label="Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Test Accuracy")
plt.legend()
plt.grid(True)

plt.show()
