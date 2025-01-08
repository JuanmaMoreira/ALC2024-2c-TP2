import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import random
from numpy.linalg import norm

def calcularLU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Args: 

        A (np.ndarray): Matriz a factorizar en LU

    Retorno:

        L (np.ndarray): Matriz triangular inferior con diagonal de unos 
            de la factorización.

        U (np.ndarray): Matriz triangular superior de la factorización.

        P (np.ndarray): Matriz de permutación. 

    La función calcula la factorización LU de la matriz A iterando por cada
    columna, asignando los coeficientes de L y triangulando para llegar a U.
    Se respeta la optimización de memoria y la optimización del pivote.
    '''
    m: int = A.shape[0]
    n: int = A.shape[1]
    Ac: np.ndarray = A.copy()
    P: np.ndarray = np.eye(n)
    
    # Chequeo de dimensión
    if m != n:
        raise AssertionError('La matriz no es cuadrada.')
        
    # Iteramos por cada columna
    for j in range(n - 1):

        # Chequeo si estamos en un sistema con ceros en la columna
        if np.allclose(Ac[j:, j], np.zeros(n - j), rtol=1e-16): break

        # Veo el máximo de la columna desde j hacía "abajo"
        idx_max: int = np.argmax(np.abs(Ac[j:, j])) + j 

        # Si mi pivote no es el máx en modulo, lo cambio para optimizar
        if j != idx_max:
            
            # Cambio filas en Ac y P
            Ac[[j, idx_max], :] = Ac[[idx_max, j], :]
            P[[j, idx_max], :] = P[[idx_max, j], :]

        # Para cada fila, modificamos los valores para obtener U
        for i in range(j + 1, m):

            # Asigno el coeficiente de L
            Ac[i][j] = Ac[i][j] /  Ac[j][j]

            # Implementación que aprovecha el formato de vector
            # actualizando columnas con la columna por el factor de la operacion
            Ac[i][j + 1 : n] -= Ac[j][j + 1: n] * Ac[i][j]         
            
    L = np.tril(Ac, -1) + np.eye(m) 
    U = np.triu(Ac)
    
    return L, U, P


def inversaLU(L: np.ndarray, U: np.ndarray, P: np.ndarray = None) -> np.ndarray:
    
    '''
    Args: 

        L (np.ndarray): Matriz triangular inferior a invertir

        U (np.ndarray): Matriz triangular superior a invertir

        P (np.ndarray): Matriz de permutación correspondiente a la 
        descomposición LU

    Retorno:

        invLU (np.ndaray): Matriz inversa de la matriz LU indicada

    Esta función utiliza las funciones inversaL e inversaU para calcular 
    la inversa de LU de una matriz aprovechando que son triangulares. Como las
    anteriores, no para de ser O(n^3) en su complejidad.
    '''

    # Calculo inversas por separado
    invL = inversaL(L)
    invU = inversaU(U)

    # Calculo la inversa del total
    invLU: np.ndarray = invU @ invL
    if not P is None: invLU = invLU @ P

    return invLU


def inversaL(L: np.ndarray) -> np.ndarray:
    '''
    Args: 

        L (np.ndarray): Matriz triangular inferior a invertir

    Retorno:

        invL (np.ndaray): Matriz triangular inferior ya invertida

    La función solo se encarga de calcular la inversa de una matriz triangular
    inferior asumiendo que está es inversible desde el vamos. La complejidad 
    es O(n^3) pero está medio oculto en las operaciones vectoriales.
    '''

    # Creo la matriz y almaceno la dimensión
    n: int = L.shape[0]
    invL: np.ndarray = np.eye(n)

    # Modifico invU de abajo hacía arriba
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            invL[j] -= (L[j][i]/L[i][i]) * invL[i]

    # Divido cada fila por el coeficiente correspondiente
    for k in range(n):
        invL[k] = invL[k] / L[k][k]            

    return invL


def inversaU(U: np.ndarray) -> np.ndarray:

    '''
    Args: 

        U (np.ndarray): Matriz triangular superior a invertir

    Retorno:

        invU (np.ndaray): Matriz triangular superior ya invertida

    La función solo se encarga de calcular la inversa de una matriz triangular
    superior asumiendo que está es inversible desde el vamos. La complejidad 
    es O(n^3) pero está medio oculto en las operaciones vectoriales.
    '''

    # Creo la matriz y almaceno la dimensión
    n: int = U.shape[0]
    invU: np.ndarray = np.eye(n)

    # Modifico invU de abajo hacía arriba
    for i in range(n - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            invU[j] -= (U[j][i]/U[i][i]) * invU[i]

    # Divido cada fila por el coeficiente correspondiente
    for k in range(n):
        invU[k] = invU[k] / U[k][k]            

    return invU

def metodoPotencia(A: np.ndarray, max_iter: int = 250, epsilon: float = 0.99999) -> np.ndarray:
    '''
    Args: 

        A (np.ndarray): Matriz a calcular mayor autovalor.

        max_iter (int): Numero máximo de operaciones

        epsilon (float): Tolerancia para el criterio de convergencia del autovector.

    Retorno:

        res (np.ndarray): Autovector de A.

    Dada una matriz, calcula el mayor autovalor de la matriz A dada como 
    argumento y retorna un float. El calculo del autovector se da a través del
    método de la potencia . 
    '''

    # Declaro un vector aleatorio de distribución uniforme
    v = random((A.shape[0], 1))
    v = v / norm(v, 2)  # Normalizar inicialmente

    iter = 0
    converge = False
    
    while iter < max_iter and not converge:
        v_next = A @ v
        v_next = v_next / norm(v_next, 2)  # Normalización
        
        # Revisar criterio de parada
        if norm(v_next - v, 2) < (1-epsilon):
            converge = True
            
        v = v_next
        iter += 1 
    
    return v


if __name__ == '__main__': 
    A1 = np.array([
        [.186, .521, .014, .32, .134],
        [.24, .073, .219, .013, .327],
        [.098, .12, .311, .302, .208],
        [.173, .03, .133, .14, .074],
        [.03, .256, .323, .225, .257]
    ])

    a1_k = np.array([metodoPotencia(A1, k=250) for _ in range(250)])
    lambda1_k = np.array(
        [(v.T @ A1 @ v)/(v.T @ v) for v in a1_k]
        )

    # Simplemente calculamos el promedio y el desvio estándar
    lambda1, lambda1_std = np.mean(lambda1_k), np.std(lambda1_k)
    print(f'Autovalor 1: {lambda1:.30f} +- {lambda1_std:.30f}')
