import numpy as np
from time import time
import matplotlib.pyplot as plt

def QR(A0):
    t1 = time()
    Ak1 = A0
    n = len(A0)
    Vk1 = np.identity(n)
    k = 0   
    betak = 1
    erro = 10**(-6)
    Autovalores = np.identity(n)
    # muk = 0
    for m in range(n-1,0,-1):
        betak = Ak1[m][m-1]
        k=0
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
            # print(Ak1)
            # print(betak)
            k+=1
        Autovalores[m][m] = Ak1[m][m]
        Ak1 = cut(Ak1,m)
        # print(Ak1)
    # Autovalores = Ak1
    Autovalores[0][0] = Ak1[0][0]
    Autovetores = Vk1
    # n = len(A0)
    # alpha = []
    # beta = []
    # for i in range(n):
    #     alpha.append(A0[i][i])
    #     if i =! n:
    #         beta.append(A0[i][i+1])
    # gamma = beta
    t2 = time()
    return Autovetores, Autovalores, t2-t1

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
def matrizA(n):
    A = np.identity(n)
    for i in range(n):
        A[i][i] = 2
        if i < n-1:
            A[i][i+1] = -1
            A[i+1][i] = -1
    return A
def matrizK(n):
    A = np.identity(n)
    m = 2
    for i in range(n):
        ki = (40+2*(i+1))/m
        ki_1 = (40+2*(i+2))/m
        A[i][i] = ki+ki_1
        if i<n-1:
            A[i][i+1] = -ki_1
            A[i+1][i] = -ki_1
    return A
n = 4
A = matrizA(4)
t_v = np.arange(0,10,0.025)
V, Lambda, t = QR(A)
# X0 = np.array([-2,-3,-1,-3,-1])
# X0 = np.transpose(X0)
X0 = np.transpose(V)[0]
print(X0)
X0 = np.transpose(X0)
Y0 = np.dot(np.transpose(V),X0)
Y = []
for i in range(len(Y0)):
    Y.append(Y0[i]*np.cos(np.sqrt(Lambda[i][i])*t_v))
Y = np.array(Y)

X = np.dot(V,Y)




for i in range(len(X0)):
    plt.figure(i)
    plt.plot(t_v,X[i])

plt.show(block=False)
plt.pause(10)
plt.close()
Lambda_analitico = []
V_analitico = []
for i in range(n):
    v = []
    Lambda_analitico.append(2*(1-np.cos((i+1)*np.pi/(n+1))))
    for j in range(n):
        v.append(np.sin((j+1)*(i+1)*np.pi/(n+1)))
    
    V_analitico.append(v)

Lambda_analitico = np.array(Lambda_analitico)
V_analitico = np.transpose(np.array(V_analitico))
print(V_analitico)



