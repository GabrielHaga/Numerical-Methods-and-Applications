import numpy as np
def delta(a):
    if a>0:
        return 1
    return -1

def calc_wi(ai):
    ei = e_vetor(len(ai))
    wi = np.transpose([ai+delta(ai[0])*norma(ai)*ei])
    return wi

def Hw(At,wi):
    wit = np.transpose(wi)
    M = At-2*np.dot(wi,np.dot(wit,At))/np.dot(wit,wi)
    b = M[0][0]
    M = np.transpose(np.transpose(M)[1:])
    # M = np.transpose(M)[1:]
    M = M-2*np.dot(M,np.dot(wi,wit))/np.dot(wit,wi)
    # print(M)
    # M = M-2*np.dot(np.dot(M,wi),wit)/np.dot(wit,wi)
    # M = np.transpose(M)
    return M,b

def calc_Hwi(H_wi_old, wi):
    n_wi = len(wi)
    n_H_wi = len(H_wi_old)
    H_wi = np.zeros([n_H_wi,n_H_wi])
    wit = np.transpose(wi)
    # print(H_wi[:n_H_wi-n_wi])
    # In = np.identity(n_wi)
    H_aux = np.transpose(H_wi_old[n_H_wi-n_wi:])
    H_aux = H_aux - 2*np.dot(H_aux,np.dot(wi,wit))/np.dot(wit,wi)
    H_wi[:n_H_wi-n_wi+1] = np.transpose(H_wi_old)[:n_H_wi-n_wi+1]
    H_wi[n_H_wi-n_wi:] = np.transpose(H_aux)
    H_wi = np.transpose(H_wi)
    # print(H_wi)
    return H_wi
    # H_wi_new =
    

def Householder(A):
    a = [A[0][0]]
    b = []
    n = len(A)
    H_wi = np.identity(n)
    At = np.copy(A)
    for i in range(n-2):
        At = At[1:]
        ai = np.transpose(At)[0]
        wi = calc_wi(ai)
        # print(wi)
        H_wi = calc_Hwi(H_wi,wi)
        At, bi = Hw(At,wi)
        a.append(At[0][0])
        b.append(bi)
        # print(At)
    a.append(At[1][1])
    b.append(At[1][0])
    a = np.array(a)
    b = np.array(b)
    # print(H_wi)
    return a, b, H_wi

def e_vetor(n):
    e = np.zeros(n)
    e[0] = 1
    return e

def norma(v):
    n_v = np.sqrt(np.dot(np.transpose(v),v))
    return n_v

def readfile_ab(file):
    A = np.loadtxt(file, dtype = 'float', delimiter = ' ',skiprows=1)
    # print(A)
    return A

def QR_espectral(a,b,c,H):
    # ti = time() # Tempo inicial para medir o tempo de simulação 

    Ck = [] # Cria a lista que contém os cossenos utilizados nas rotações de Givens
    Sk = [] # Cria a lista que contém os senos utilizados nas rotações de Givens

    # Vetores auxiliares utilizados para o cálculo dos vetores principais
    aj = []
    bj = []
    cj = []

    erro_max = 1e-6 # Estabelece o erro máximo para o critério de parada
    iteracao = 0 # Variável que conta o número de iterações
    n = len(a) # Ordem da matriz tridiagonal
    autovalores = np.zeros(n) # Inicializa o vetor que contém os autovalores
    muk = 0 # Inicializa o fator calculado na heurística
    # V = np.identity(n) # Inicializa a matriz do autovalores 
    V = np.copy(H)
    # Implementação do algoritmo
    for m in range(n-1,0,-1):
        erro = np.abs(b[m-1]) # Define o erro como o beta_(m-1), pois ele deve
        # assumir um valor próximo de 0 quando alpha_(m-1) converge para um autovalor
        while erro > erro_max: # Define o critério de parada
            # Heurística de Wilkinson
            if iteracao > 0: # Condicional para entrar na heurística de Wilkinson
                dk = (a[m-1]-a[m])/2
                muk = a[m] + dk-np.sign(dk)*(dk**2 + b[m-1]**2)**(1/2)
                a = a-muk*np.ones(m+1) # Tira muk dos valores da diagonal principal
            # "Calculo do R"
            for i in range(len(a)-1):
                # Calculo dos senos e cossenos da rotação de Givens
                if abs(a[i]) > abs(b[i]):
                    tau = -b[i]/a[i]
                    ck = 1/(1+tau**2)**(1/2)
                    sk = ck*tau
                    Ck.append(ck)
                    Sk.append(sk)
                else:
                    tau = -a[i]/b[i]
                    sk = 1/(1+tau**2)**(1/2)
                    ck = sk*tau
                    Ck.append(ck)
                    Sk.append(sk)
                Qki = calc_Qki(i,n,ck,sk) # Matriz da rotação de Givens de ordem n
                V = np.dot(V,np.transpose(Qki)) # Calculo da matriz dos autovetores

                # Aplica a rotação de Givens nas linhas i e i+1 
                aj.append([Ck[i]*a[i] - Sk[i]*b[i], Sk[i]*c[i] + Ck[i]*a[i+1]])
                cj.append([Ck[i]*c[i] - Sk[i]*a[i+1], Ck[i]*c[i+1]])
                bj.append(Sk[i]*a[i] + Ck[i]*b[i])

                # Guarda os resultados nos vetores alpha, beta e gamma
                a[i] = aj[i][0]
                c[i] = cj[i][0]
                b[i] = bj[i]
                a[i+1] = aj[i][1]
                c[i+1] = cj[i][1]

            # 'Calculo do A^(k+1)'
            for i in range(len(a)-1):
                a[i] = a[i]*Ck[i] - c[i]*Sk[i]
                b[i] = -a[i+1]*Sk[i]
                c[i] = b[i]
                a[i+1] = a[i+1]*Ck[i]

            erro = np.abs(b[m-1]) # 'Recalcula' o erro, pois o beta[m-1] foi atualizado  
            iteracao = iteracao + 1 # Atualiza o número da iteração

            # Esvazia as listas auxiliares
            aj = []
            bj = []
            cj = []
            Ck = []
            Sk = [] 

            a = a + muk*np.ones(m+1) # Soma de volta muk na diagonal principal
        autovalores[m] = a[m] # Guarda o autovalor calculado para esse m

        # Redução do tamanho das matrizes
        a = a[:m] 
        b = b[:m-1]
        c = c[:m]

    # tf = time() # Tempo final

    autovalores[0] = a[0] # Guarda o valor do maior autovalor

    return V, autovalores

# Função que calcula a matriz rotação de Givens
def calc_Qki(i, n, ck, sk):
    Qki = np.identity(n) # Inicializa a matriz Q_i^(k+1) com a identidade
    # Substitui pelos senos e cossenos para tonar na matriz rotação de Givens
    Qki[i][i] = ck 
    Qki[i+1][i+1] = ck
    Qki[i][i+1] = -sk
    Qki[i+1][i] = sk
    return Qki

    
#Lê arquivo input-c
def readfile_c(file):
    dados = np.loadtxt(file, dtype = 'float', delimiter = ' ',max_rows=2)
    # print(dados)   
    barras = np.loadtxt(file, dtype = 'float', delimiter = ' ',skiprows=2)
    # print(barras)
    return dados, barras

def MatrizC():
    dados, barras = readfile_c('input-c')
    n = int(2*dados[0][1]) # número de nós não fixos
    rho = dados[1][0] # massa específica do material [kg/m^3]
    A = dados[1][1] # área [m^2]
    E = dados[1][2] # módulo de elasticidade [GPa]

    # Construção da matriz de massa
    M = np.zeros([n,n])
    M12 = np.zeros([n,n])

    for index, i in enumerate(barras[:,0]):
        M[2*int(i)-2,2*int(i)-2] += 0.5*rho*A*barras[index,3]
        M[2*int(i)-1,2*int(i)-1] += 0.5*rho*A*barras[index,3]

    for index, i in enumerate(barras[:,1]):
        if i < n/2 + 1:
            M[2*int(i)-2,2*int(i)-2] += 0.5*rho*A*barras[index,3]
            M[2*int(i)-1,2*int(i)-1] += 0.5*rho*A*barras[index,3]

    for i in range(n):
        M12[i][i] = M[i][i]**(-1/2)
        # print(M12[i][i])    

    # Construção da matriz de rigidez
    K = np.zeros([n,n])

    # Percorre as colunas da matriz que contêm dos dados da barra
    for i,j,k,l in barras: # i e j - nós; k = ângulo [°]; l = comprimento da barra [m]
        # if j < n/2 + 1:
        # coluna 1
        K[2*int(i)-2][2*int(i)-2] += A*E*10**9/l*(np.cos(k*np.pi/180))**2
        K[2*int(i)-1][2*int(i)-2] += A*E*10**9/l*(np.cos(k*np.pi/180))*(np.sin(k*np.pi/180))
        K[2*int(i)-2][2*int(i)-1] += A*E*10**9/l*(np.cos(k*np.pi/180))*(np.sin(k*np.pi/180))
        K[2*int(i)-1][2*int(i)-1] += A*E*10**9/l*(np.sin(k*np.pi/180))**2
        if j<n/2+1:
            K[2*int(j)-2][2*int(i)-2] += - A*E*10**9/l*(np.cos(k*np.pi/180))**2
            K[2*int(j)-1][2*int(i)-2] += - A*E*10**9/l*(np.cos(k*np.pi/180))*(np.sin(k*np.pi/180))

            # # coluna 2
            # K[2*int(i)-2,2*int(i)-1] += A*E*10**9/l*(np.cos(k*np.pi/180))*(np.sin(k*np.pi/180))
            # K[2*int(i)-1,2*int(i)-1] += A*E*10**9/l*(np.sin(k*np.pi/180))**2
            K[2*int(j)-2][2*int(i)-1] += - A*E*10**9/l*(np.cos(k*np.pi/180))*(np.sin(k*np.pi/180))
            K[2*int(j)-1][2*int(i)-1] += - A*E*10**9/l*(np.sin(k*np.pi/180))**2

            # coluna 3
            K[2*int(i)-2][2*int(j)-2] += - A*E*10**9/l*(np.cos(k*np.pi/180))**2
            K[2*int(i)-1][2*int(j)-2] += - A*E*10**9/l*(np.cos(k*np.pi/180))*(np.sin(k*np.pi/180))
            K[2*int(j)-2][2*int(j)-2] += A*E*10**9/l*(np.cos(k*np.pi/180))**2
            K[2*int(j)-1][2*int(j)-2] += A*E*10**9/l*(np.cos(k*np.pi/180))*(np.sin(k*np.pi/180))

            # coluna 4
            K[2*int(i)-2][2*int(j)-1] += - A*E*10**9/l*(np.cos(k*np.pi/180))*(np.sin(k*np.pi/180))
            K[2*int(i)-1][2*int(j)-1] += - A*E*10**9/l*(np.sin(k*np.pi/180))**2
            K[2*int(j)-2][2*int(j)-1] += A*E*10**9/l*(np.cos(k*np.pi/180))*(np.sin(k*np.pi/180))
            K[2*int(j)-1][2*int(j)-1] += A*E*10**9/l*(np.sin(k*np.pi/180))**2
    A = np.dot(M12,np.dot(K,M12))
    return A, M

def main():
    
    # Salva a matriz de rigidez em arquivo externo
    # np.savetxt("K2.csv", K, delimiter=" , ")
    # np.savetxt("M.csv", M, delimiter=" , ")
    # A = np.array([[2,-1,1,3],[-1,1,4,2],[1,4,2,-1],[3,2,-1,1]])
    # A = readfile_ab('input-a')
    # print(A)
    A,M = MatrizC()
    # print(A)
    a,b,H = Householder(A)
    print(a)
    c = np.zeros(len(a))
    c[:len(b)] = np.copy(b)
    V, Autovalores = QR_espectral(a,b,c,H)
    print(np.sqrt(Autovalores))
    # print(V)
    # print(b)
    # print(c)
    # At = np.transpose(A[1:])



