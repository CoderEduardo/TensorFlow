import tensorflow as tf

#tensores: conceito básico, um tensor é uma matriz multidimensional

#criando tensores
#escalar 0d
scalar = tf.constant(5)
print(f"Escalar: {scalar}")

#vetor 1d
vector = tf.constant([1,2,3])
print(f"Vector 1d: {vector}")

#matriz 2d
matriz = tf.constant([[1,2],[3,4]])
print(f"Matriz 2d: {matriz}")

#tensor 3d
tensor3d = tf.constant([[[1],[2],[3]]])
print(f"Tensor 3d: {tensor3d}")

#operações com tensores 
a = tf.constant([1,2,3])
b= tf.constant([4,5,6])

#soma 
print("Soma: ", tf.add(a, b))

#multiplicação elemento a elemento
print(f"Multiplicação: {tf.multiply(a,b)}")

#produto escalar 
print(f"Produto escalar: {tf.tensordot(a,b, axes=1)}")

#Variáveis 
#Tensores constantes não mudam. Para treinamento de redes neurais, usamos tf.Variable:
x = tf.Variable([1.0,2.0,3.0])
x.assign([4.0,5.0,6.0])     #atualizando valores
print(x)