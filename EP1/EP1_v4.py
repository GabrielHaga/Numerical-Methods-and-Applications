import numpy as np
from time import time
import matplotlib.pyplot as plt

# Função que implementa algoritmo QR com deslocamentos espectrais 
# A entrada são os vetores alpha, beta e gamma, representados pelas letras a, b e c, respectivamente
def QR_espectral(a,b,c):
    ti = time() # Tempo inicial para medir o tempo de simulação 

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
    V = np.identity(n) # Inicializa a matriz do autovalores 

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

    tf = time() # Tempo final

    autovalores[0] = a[0] # Guarda o valor do maior autovalor

    return V, autovalores, iteracao, tf-ti

# Função que implementa algoritmo QR sem deslocamentos espectrais 
# A entrada são os vetores alpha, beta e gamma, representados pelas letras a, b e c, respectivamente
def QR(a,b,c):
    ti = time() # Tempo inicial para medir o tempo de simulação 

    Ck = [] # Cria a lista que contém os cossenos utilizados nas rotações de Givens
    Sk = [] # Cria a lista que contém os senos utilizados nas rotações de Givens

    # Vetores auxiliares utilizados para o cálculo dos vetores principais
    aj = []
    bj = []
    cj = []

    erro = 1 # Inicializa o valor do erro como 1
    erro_max = 1e-6 # Estabelece o erro máximo para o critério de parada
    iteracao = 0 # Variável que conta o número de iterações
    n = len(a) # Ordem da matriz tridiagonal
    V = np.identity(n) # Inicializa a matriz do autovalores 

    # Implementação do algoritmo QR
    while erro > erro_max: # Define o critério de parada
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
            V = np.dot(V,np.transpose(Qki))# Calculo da matriz dos autovetores

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
        
 
        erro = max(np.abs(b)) # 'Recalcula' o erro, pois o beta foi atualizado  
        iteracao = iteracao + 1 # Atualiza o número da iteração

        # Esvazia as listas auxiliares
        aj = []
        bj = []
        cj = []
        Ck = []
        Sk = [] 
    tf = time() # Tempo final

    return V, a, iteracao, tf-ti

# Função que calcula a matriz rotação de Givens
def calc_Qki(i, n, ck, sk):
    Qki = np.identity(n) # Inicializa a matriz Q_i^(k+1) com a identidade
    # Substitui pelos senos e cossenos para tonar na matriz rotação de Givens
    Qki[i][i] = ck 
    Qki[i+1][i+1] = ck
    Qki[i][i+1] = -sk
    Qki[i+1][i] = sk
    return Qki

# Função que cria os vetores alpha, beta e gamma, representados pelas letras a, b e c, respectivamente
def calc_abc(n, opcoes):
    # Divide nos itens (a), (b) e (c), definindo o item através da variável 'opcoes'

    # Item (a)
    if opcoes == 'a':
        a = np.full(n,2).tolist()
        b = np.full(n-1,-1).tolist()
        c = np.full(n-1,-1).tolist()
        # b.append(0)
        c.append(0)

    # Item (b)
    elif opcoes == 'b':
        m = 2
        a = np.zeros(n)
        b = np.zeros(n-1)
        c = np.zeros(n)
        for i in range(n):
            ki = (40+2*(i+1))/m # Calcula a constante de rigidez da mola i
            ki_1 = (40+2*(i+2))/m # Calcula a constante de rigidez da mola i+1
            a[i] = ki+ki_1
            if i<n-1:
                b[i] = -ki_1
                c[i] = -ki_1

    # Item (c)
    elif opcoes == 'c':
        m = 2
        a = np.zeros(n)
        b = np.zeros(n-1)
        c = np.zeros(n)
        for i in range(n):
            ki = (40+2*(-1)**(i+1))/m # Calcula a constante de rigidez da mola i
            ki_1 = (40+2*(-1)**(i+2))/m # Calcula a constante de rigidez da mola i+1
            a[i] = ki+ki_1
            if i<n-1:
                b[i] = -ki_1
                c[i] = -ki_1
    return a, b, c

# Função principal do código
def main():

    print('Os itens válidos são: (a), (b) ou (c): ')
    item = input('Escolha o item a ser resolvido [a-Item (a), b-Item (b), c-Item (c)]: ')
    while item != 'a' and item != 'b' and item != 'c':
        print('Escolha um item válido')
        item = input('Escolha o item a ser resolvido: ')


    if item == 'a':
        print('Os valores válidos para n são: 4, 8, 16 e 32')
        n = input('Escolha o n: ')
        while n != '4' and n != '8' and n != '16' and n != '32':
            print('Escolha um n válido')
            n = input('Escolha o n: ')
        n = int(n)
        a, b, c = calc_abc(n, item)
        tipo_QR = input('Escolha qual o algoritmo [1-Sem deslocamento, 2-Com deslocamento]:')
        while tipo_QR != '1' and tipo_QR != '2':
            print('Escolha uma das opções')
            tipo_QR = input('Escolha qual o algoritmo [1-Sem deslocamento, 2-Com deslocamento]:')
        
        if tipo_QR == '2':
            V, Lambda, iter, t = QR_espectral(a,b,c)
            print('\nAlgoritmo QR com deslocamento espectrais',',Item',item,', n =', n)
        elif tipo_QR == '1':
            V, Lambda, iter, t = QR(a,b,c)
            print('\nAlgoritmo QR sem deslocamento espectrais',',Item',item,', n =', n)
        print('\nNúmero de iterações :\n', iter)
        print('\nAutovalores :\n', Lambda)
        print('\nAutovetores :\n', V)

    elif item == 'b':
        n = 5
        ter_graficos = input('Deseja ver os gráficos [s-sim, n-não]: ')
        while ter_graficos != 's' and ter_graficos != 'n':
            print('Escolha uma opção válida')
            ter_graficos = input('Deseja ver os gráficos [s-sim, n-não]: ')
        a, b, c = calc_abc(n, item)
        tipo_QR = input('Escolha qual o algoritmo [1-Sem deslocamento, 2-Com deslocamento]:')
        while tipo_QR != '1' and tipo_QR != '2':
            print('Escolha uma das opções')
            tipo_QR = input('Escolha qual o algoritmo [1-Sem deslocamento, 2-Com deslocamento]:')
        
        print('Caso 1: X(0) = -2, -3, -1, -3, -1', '\nCaso 2: X(0) = 1, 10, -4, 3, -2', '\nCaso 3: X(0) correspondente ao modo de maior frequência')
        caso = input('Escolha o caso [1-Caso 1, 2-Caso 2, 3-Caso 3]: ')
        while caso != '1' and caso != '2' and caso != '3':
            print('Escolha um caso válido')
            caso = input('Escolha o caso [1-Caso 1, 2-Caso 2, 3-Caso 3]: ')
        caso = int(caso)
        if tipo_QR == '2':
            V, Lambda, iter, t = QR_espectral(a,b,c)
            print('\nAlgoritmo QR com deslocamento espectrais',',Item',item,', caso:', caso, 'n =',n)
        elif tipo_QR == '1':
            V, Lambda, iter, t = QR(a,b,c)
            print('\nAlgoritmo QR sem deslocamento espectrais',',Item',item,', caso:', caso, 'n =',n)
        if ter_graficos == 's':
            if caso == 1:
                X0 = np.transpose(np.array([-2, -3, -1, -3, -1]))
            elif caso == 2:
                X0 = np.transpose(np.array([1, 10, -4, 3, -2]))
            else:
                X0 = np.transpose(np.transpose(V)[0])
            t_v = np.arange(0,10,0.025)
            Y0 = np.dot(np.transpose(V),X0)
            Y = []
            for i in range(len(Y0)):
                Y.append(Y0[i]*np.cos(np.sqrt(Lambda[i])*t_v))
            Y = np.array(Y)
            X = np.dot(V,Y)
            for i in range(len(X0)):
                plt.figure(i)
                plt.plot(t_v,X[i])
        print('Item:',item, ', caso:', caso, 'n =',n)
        print('\nNúmero de iterações :\n', iter)
        print('\nAutovalores :\n', Lambda)
        print('\nAutovetores :\n', V)
        plt.show()
    else:
        n = 10
        ter_graficos = input('Deseja ver os gráficos [s-sim, n-não]: ')
        a, b, c = calc_abc(n, item)
        tipo_QR = input('Escolha qual o algoritmo [1-Sem deslocamento, 2-Com deslocamento]:')
        while tipo_QR != '1' and tipo_QR != '2':
            print('Escolha uma das opções')
            tipo_QR = input('Escolha qual o algoritmo [1-Sem deslocamento, 2-Com deslocamento]:')
        
        print('Caso 1: X(0) = -2, -3, -1, -3, -1, -2, -3, -1, -3, -1', '\nCaso 2: X(0) = 1, 10, -4, 3, -2, 1, 10, -4, 3, -2', '\nCaso 3: X(0) correspondente ao modo de maior frequência')
        caso = input('Escolha o caso [1-Caso 1, 2-Caso 2, 3-Caso 3]: ')
        while caso != '1' and caso != '2' and caso != '3':
            print('Escolha um caso válido')
            caso = input('Escolha o caso [1-Caso 1, 2-Caso 2, 3-Caso 3]: ')
        caso = int(caso)
        if tipo_QR == '2':
            V, Lambda, iter, t = QR_espectral(a,b,c)
            print('\nAlgoritmo QR com deslocamento espectrais',',Item',item,', caso:', caso, 'n =',n)
        elif tipo_QR == '1':
            V, Lambda, iter, t = QR(a,b,c)
            print('\nAlgoritmo QR sem deslocamento espectrais',',Item',item,', caso:', caso, 'n =',n)
        
        if ter_graficos == 's':
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
                Y.append(Y0[i]*np.cos(np.sqrt(Lambda[i])*t_v))
            Y = np.array(Y)
            X = np.dot(V,Y)
            for i in range(len(X0)):
                plt.figure(i)
                plt.plot(t_v,X[i])
        print('\nNúmero de iterações :\n', iter)
        print('\nAutovalores :\n', Lambda)
        print('\nAutovetores :\n', V)
        plt.show()
main()