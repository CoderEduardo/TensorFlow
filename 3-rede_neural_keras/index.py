import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

#rede neural que aprender a reconhecer digitos manuscritos

#carregar dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalizar valores (0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Transformar em vetor (flatten)
#Transforma imagens 28x28 em vetores de 784 elementos
x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

print("Formato de treino:", x_train.shape)
print("Formato de teste:", x_test.shape)

"""Criando a Rede Neural
Usaremos um modelo Sequencial.
Estrutura:
Flatten → transforma imagem em vetor.
Dense (128 neurônios, ReLU) → camada oculta.
Dense (10 neurônios, Softmax) → saída (0 a 9)."""

model = models.Sequential([          #modelo linear, camadas em sequência
    layers.Input(shape=(784,)),     #entrada (28x28 pixels achatados)
    layers.Dense(128, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])

#compilando o modelo
model.compile(
    optimizer='adam',                           #otimizador eficiente para ajustes de pesos
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']                        #metrica de avaliação
)

model.summary()

#treinando o modelo
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),   #avalia performace no teste durante o treino
    epochs=5,                           #treina por 5 épocas, passagens completas pelo database
    batch_size=32                       #atualiza pesos a cada 32 exemplos
)

#avaliando o modelo
loss, acc = model.evaluate(x_test, y_test)
print(f"Acurácia no teste: {acc*100:.2f}%")

#visualizando o treinamento 
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

#testando previsões
img = x_test[0].reshape(1,784)
pred = model.predict(img)

print(f'Probabilidades: {pred}')
print(f'Digital previsto: {np.argmax(pred)}')
print(f"Digita real: {y_test[0]}")