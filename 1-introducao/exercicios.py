import tensorflow as tf

#Crie os seguintes tensores:

#Escalar com valor 10
escalar = tf.constant(10)
print(escalar)

#Vetor com valores [10, 20, 30, 40]
vetor = tf.constant([10,20,30,40])
print(vetor)

#Matriz 3x3 com números de 1 a 9
matriz = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
print(matriz)

#Faça as operações:

#Soma e multiplicação elemento a elemento de dois vetores.
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
print(f"Soma: {tf.add(a,b)}")
print(f"Multiplicação: {tf.multiply(a,b)}")

#Produto escalar de dois vetores.
print(f"Produto escalar: {tf.tensordot(a,b,axes=1)}")

#Transforme um tensor 1D em matriz 2x2 usando reshape.
d2 = tf.reshape(vetor,[2,2])
print(d2)

#Crie uma tf.Variable e altere seus valores com .assign().
variavel = tf.Variable([90,80,70])
print(variavel)
variavel.assign([70,80,90])
print(variavel)