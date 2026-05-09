import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- MNIST podatkovni skup ---
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikaži nekoliko slika iz train skupa 
plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Labela: {y_train[i]}")
    plt.axis('off')
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1] 
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa 
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one-hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

# TODO: kreiraj mrezu pomocu keras.Sequential() 
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile() 
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# TODO: provedi treniranje mreze pomocu .fit() 
model.fit(x_train_s, y_train_s, epochs=25, batch_size=32, validation_split=0.1)

# TODO: Izracunajte tocnost mreze na skupu za učenje i testiranje 
score_train = model.evaluate(x_train_s, y_train_s, verbose=0)
score_test = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Tocnost (train): {score_train[1]:.4f}")
print(f"Tocnost (test): {score_test[1]:.4f}")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje 
y_test_pred = model.predict(x_test_s)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

cm = confusion_matrix(y_test, y_test_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrica zabune")
plt.show()

misclassified_idx = np.where(y_test_pred_classes != y_test)[0]
plt.figure(figsize=(10, 4))
for i, idx in enumerate(misclassified_idx[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Stvarno: {y_test[idx]}\nPred.: {y_test_pred_classes[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()