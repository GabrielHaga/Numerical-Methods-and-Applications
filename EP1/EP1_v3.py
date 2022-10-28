import numpy as np
from time import time
import matplotlib.pyplot as plt

def QR_espectral(A0):
    t1 = time()
    Ak1 = A0
    n = len(A0)
    Vk1 = np.identity(n)
    k = 0
    betak = 1
    erro = 10**(-6)
    Autovalores = np.identity(n)
    for m in range(n-1,0,-1):
        betak = Ak1[m][m-1]
        muk = 0
        while abs(betak)>erro:
            I = np.identity(m+1)
            Ak = Ak1
            Vk = Vk1
            Qk = I
            if k>0:
                alphak_n1 = Ak[m-1][m-1]
                alphak_n = Ak[m][m]
                betak_n1 = Ak[m][m-1]
                dk = (alphak_n1-alphak_n)/2
                muk = alphak_n + dk-np.sign(dk)*(dk**2 + (betak_n1)**2)**(1/2)
                # print(muk)
            Rk = Ak - muk*I
            for j in range(m):
                Rj,Qj = Givens(j,Rk)
                Qj = np.transpose(Qj)
                Qk = np.dot(Qk,Qj)
                Rk = Rj
            Ak1 = np.dot(Rk,Qk)+muk*I
            if m == n-1:
                Qk_V = Qk
            else:
                Qk_V = calc_Qk(Qk,n)
            Vk1 = np.dot(Vk,Qk_V)
            betak = Ak1[m][m-1]
            k+=1
        Autovalores[m][m] = Ak1[m][m]
        Ak1 = cut(Ak1,m)
    Autovalores[0][0] = Ak1[0][0]
    Autovetores = Vk1
    
    t2 = time()
    return Autovetores, Autovalores, k, t2-t1

def QR(A0):
    t1 = time()
    Ak1 = A0
    n = len(A0)
    I = np.identity(n)
    Vk1 = I
    k = 0   
    erro = 1
    erro_max = 10**(-6)
    k=0
    while erro>erro_max:
        Ak = Ak1
        Vk = Vk1
        Qk = I
        Rk = Ak 
        for j in range(n-1):
            Rj,Qj = Givens(j,Rk)
            Qj = np.transpose(Qj)
            Qk = np.dot(Qk,Qj)
            Rk = Rj
            # print(Qj)
        Ak1 = np.dot(Rk,Qk)
        Vk1 = np.dot(Vk,Qk)
        erro = np.abs(Ak1[1][0])
        for i in range(len(Ak1)-2):
            if erro < np.abs(Ak1[i+2][i+1]):
                erro = np.abs(Ak1[i+2][i+1])
        k+=1
    Autovalores = np.identity(n)
    for i in range(n):
        Autovalores[i][i] = Ak1[i][i]
    Autovetores = Vk1
    t2 = time()
    return Autovetores, Autovalores,k,t2-t1

def calc_Qk(Qk,n):
    l = len(Qk)
    Qk_V = np.identity(n)
    for i in range(l):
        for j in range(l):
            Qk_V[i][j] = Qk[i][j]
    return Qk_V

def cut(A,m):
    B = np.identity(m)
    for i in range(m):
        for j in range(m):
            B[i][j] = A[i][j]
    return B

def Givens(i,A):
    alpha = A[i][i]
    beta = A[i+1][i]
    n = len(A)
    if abs(alpha)>abs(beta):
        tau = -beta/alpha
        c = 1/np.sqrt(1+tau**2)
        s = c*tau
    else:
        tau = -alpha/beta
        s = 1/np.sqrt(1+tau**2)
        c = s*tau
    Q = np.identity(n)
    Q[i][i] = c
    Q[i][i+1] = -s
    Q[i+1][i] = s
    Q[i+1][i+1] = c
    R = np.dot(Q,A)
    return R,Q

def matrizA(n, opcoes):
    A = np.identity(n)
    if opcoes == 'a':
        for i in range(n):
            A[i][i] = 2
            if i < n-1:
                A[i][i+1] = -1
                A[i+1][i] = -1

    elif opcoes == 'b':
        m = 2
        for i in range(n):
            ki = (40+2*(i+1))/m
            ki_1 = (40+2*(i+2))/m
            A[i][i] = ki+ki_1
            if i<n-1:
                A[i][i+1] = -ki_1
                A[i+1][i] = -ki_1

    elif opcoes == 'c':
        m = 2
        for i in range(n):
            ki = (40+2*(-1)**(i+1))/m
            ki_1 = (40+2*(-1)**(i+2))/m
            A[i][i] = ki+ki_1
            if i<n-1:
                A[i][i+1] = -ki_1
                A[i+1][i] = -ki_1
    return A

def main():

    print('Os itens válidos são a, b ou c: ')
    item = input('Escolha o item a ser resolvido: ')
    while item != 'a' and item != 'b' and item != 'c':
        print('Escolha um item válido')
        item = input('Escolha o item a ser resolvido: ')


    if item == 'a':
        print('Os valores válidos para n são 4, 8, 16 e 32!')
        n = input('Escolha o n: ')
        # while n != '4' and n != '8' and n != '16' and n != '32':
        #     print('Escolha um n válido')
        #     n = input('Escolha o n: ')
        n = int(n)
        A = matrizA(n,item)
        V, Lambda, iter, t = QR_espectral(A)
        # V2, Lambda2, iter2, t2 = QR(A)
        print(t)
        # print('Número de iterações para o algoritmo QR:', iter2, '\nNúmero de iterações para o algoritmo QR com deslcamentos espectrais:', iter)
        # print('Autovalores calculado com o algoritmo QR:', Lambda2, '\nAutovalores calculado com o algoritmo QR com deslcamentos espectrais:', Lambda)
        # print('Autovetores calculado com o algoritmo QR:', V2, '\nAutovetores calculado com o algoritmo QR com deslcamentos espectrais:', V)

    elif item == 'b':
        n = 5
        print('Caso 1: X(0) = -2, -3, -1, -3, -1', '\nCaso 2: X(0) = 1, 10, -4, 3, -2', '\nCaso 3: X(0) correspondente ao modo de maior frequência')
        # print('Para escolher utilize o número do caso. Por exemplo se for o caso 1, você deve colocar como entrada o número 1')
        caso = input('Escolha o caso: ')
        while caso != '1' and caso != '2' and caso != '3':
            print('Escolha um caso válido')
            caso = input('Escolha o caso: ')
        caso = int(caso)
        A = matrizA(n,item)
        V, Lambda, iter, t = QR_espectral(A)
        if caso == 1:
            X0 = np.transpose(np.array([-2, -3, -1, -3, -1]))
        elif caso == 2:
            X0 = np.transpose(np.array([1, 10, -4, 3, -2]))
        else:
            X0 = np.transpose(np.transpose(V)[0])
        t_v = np.arange(0,10,0.025)
        Y0 = np.dot(np.transpose(V),X0)
        print(Y0[0])
        Y = []
        for i in range(len(Y0)):
            Y.append(Y0[i]*np.cos(np.sqrt(Lambda[i][i])*t_v))
        Y = np.array(Y)
        X = np.dot(V,Y)
        for i in range(len(X0)):
            plt.figure(i)
            plt.plot(t_v,X[i])
        plt.show()
        # plt.show(block=False)
        # plt.pause(10)
        # plt.close()
    else:
        n = 10
        print('Caso 1: X(0) = -2, -3, -1, -3, -1, -2, -3, -1, -3, -1', '\nCaso 2: X(0) = 1, 10, -4, 3, -2, 1, 10, -4, 3, -2', '\nCaso 3: X(0) correspondente ao modo de maior frequência')
        caso = input('Escolha o caso: ')
        while caso != '1' and caso != '2' and caso != '3':
            print('Escolha um caso válido')
            caso = input('Escolha o caso: ')
        caso = int(caso)
        A = matrizA(n,item)
        V, Lambda, iter, t = QR_espectral(A)
        print(V)
        print(Lambda)
        if caso == 1:
            X0 = np.transpose(np.array([-2, -3, -1, -3, -1, -2, -3, -1, -3, -1]))
        elif caso == 2:
            X0 = np.transpose(np.array([1, 10, -4, 3, -2, 1, 10, -4, 3, -2]))
        else:
            X0 = np.transpose(np.transpose(V)[0])
        t_v = np.arange(0,10,0.025)
        Y0 = np.dot(np.transpose(V),X0)
        Y = []
        for i in range(len(Y0)):
            Y.append(Y0[i]*np.cos(np.sqrt(Lambda[i][i])*t_v))
        Y = np.array(Y)
        X = np.dot(V,Y)
        for i in range(len(X0)):
            plt.figure(i)
            plt.plot(t_v,X[i])
        plt.show()
main()
# a = np.array([1,2,3,4,5,6,7])
# a = a[:4]
