import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np

#Dataset MNIST

"""O MNIST é um dataset clássico com imagens de dígitos manuscritos (0 a 9).
Contém 60.000 imagens para treino e 10.000 para teste.z'
Cada imagem é 28x28 pixels, em escala de cinza."""

#carregar o dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#verificar dimensões
print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_test: {y_test.shape}")

#normalizando dados:    redes neurais funcionam melhor com dados entre 0 e 1

#convertendo para float32 e normalizando
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#visualizando as imagens

#mostrando a primeira imagem de treino
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show() 

#mostrar 9 imagens
plt.figure(figsize=(6,6))
for i in range (9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()