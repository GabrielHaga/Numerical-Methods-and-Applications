import numpy as np
import matplotlib.pyplot as plt
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
    # H_wi = np.zeros([n_H_wi,n_H_wi])

    # # print(H_wi[:n_H_wi-n_wi])
    # # In = np.identity(n_wi)
    # H_aux = np.transpose(H_wi_old[n_H_wi-n_wi:])
    # H_aux = H_aux - 2*np.dot(np.dot(H_aux ,wi),wit)/np.dot(wit,wi)
    # # print(np.dot(wi,wit))
    # H_wi[:n_H_wi-n_wi+1] = np.transpose(H_wi_old)[:n_H_wi-n_wi+1]
    # H_wi[n_H_wi-n_wi:] = np.transpose(H_aux)
    # H_wi = np.transpose(H_wi)

    H_aux = np.identity(n_H_wi)
    wit = np.transpose(wi)
    H_wi_r = np.identity(n_wi)-2*np.dot(wi,wit)/np.dot(wit,wi)
    for i in range(n_wi):

        H_aux[i+n_H_wi-n_wi][n_H_wi-n_wi:] = H_wi_r[i]
    H_wi = np.dot(H_wi_old,H_aux)
    return H_wi
    
def calc_Hwi_v1(H_wi_old, wi):

    n_wi = len(wi)
    n_H_wi = len(H_wi_old)
    H_wi = np.zeros([n_H_wi,n_H_wi])
    wit = np.transpose(wi)
    # print(H_wi[:n_H_wi-n_wi])
    # In = np.identity(n_wi)
    H_aux = np.transpose(H_wi_old[n_H_wi-n_wi:])
    H_aux = H_aux - 2*np.dot(np.dot(H_aux ,wi),wit)/np.dot(wit,wi)
    # print(np.dot(wi,wit))
    H_wi[:n_H_wi-n_wi+1] = np.transpose(H_wi_old)[:n_H_wi-n_wi+1]
    H_wi[n_H_wi-n_wi:] = np.transpose(H_aux)
    H_wi = np.transpose(H_wi)
    return H_wi

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
    # H = np.transpose(H_wi)
    # print(H_wi)
    return a, b, H_wi

def e_vetor(n):
    e = np.zeros(n)
    e[0] = 1
    return e

def norma(v):
    n_v = np.sqrt(np.dot(np.transpose(v),v))
    return n_v


def QR_espectral(a,b,c,H):
    # ti = time() # Tempo inicial para medir o tempo de simula????o 

    Ck = [] # Cria a lista que cont??m os cossenos utilizados nas rota????es de Givens
    Sk = [] # Cria a lista que cont??m os senos utilizados nas rota????es de Givens

    # Vetores auxiliares utilizados para o c??lculo dos vetores principais
    aj = []
    bj = []
    cj = []

    erro_max = 1e-10 # Estabelece o erro m??ximo para o crit??rio de parada
    iteracao = 0 # Vari??vel que conta o n??mero de itera????es
    n = len(a) # Ordem da matriz tridiagonal
    autovalores = np.zeros(n) # Inicializa o vetor que cont??m os autovalores
    muk = 0 # Inicializa o fator calculado na heur??stica
    # V = np.identity(n) # Inicializa a matriz do autovalores 
    V = np.copy(H)
    # Implementa????o do algoritmo
    for m in range(n-1,0,-1):
        erro = np.abs(b[m-1]) # Define o erro como o beta_(m-1), pois ele deve
        # assumir um valor pr??ximo de 0 quando alpha_(m-1) converge para um autovalor
        while erro > erro_max: # Define o crit??rio de parada
            # Heur??stica de Wilkinson
            if iteracao > 0: # Condicional para entrar na heur??stica de Wilkinson
                dk = (a[m-1]-a[m])/2
                muk = a[m] + dk-np.sign(dk)*(dk**2 + b[m-1]**2)**(1/2)
                a = a-muk*np.ones(m+1) # Tira muk dos valores da diagonal principal
            # "Calculo do R"
            for i in range(len(a)-1):
                # Calculo dos senos e cossenos da rota????o de Givens
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
                Qki = calc_Qki(i,n,ck,sk) # Matriz da rota????o de Givens de ordem n
                V = np.dot(V,np.transpose(Qki)) # Calculo da matriz dos autovetores

                # Aplica a rota????o de Givens nas linhas i e i+1 
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
            iteracao = iteracao + 1 # Atualiza o n??mero da itera????o

            # Esvazia as listas auxiliares
            aj = []
            bj = []
            cj = []
            Ck = []
            Sk = [] 

            a = a + muk*np.ones(m+1) # Soma de volta muk na diagonal principal
        autovalores[m] = a[m] # Guarda o autovalor calculado para esse m

        # Redu????o do tamanho das matrizes
        a = a[:m] 
        b = b[:m-1]
        c = c[:m]

    # tf = time() # Tempo final

    autovalores[0] = a[0] # Guarda o valor do maior autovalor

    return V, autovalores

# Fun????o que calcula a matriz rota????o de Givens
def calc_Qki(i, n, ck, sk):
    Qki = np.identity(n) # Inicializa a matriz Q_i^(k+1) com a identidade
    # Substitui pelos senos e cossenos para tonar na matriz rota????o de Givens
    Qki[i][i] = ck 
    Qki[i+1][i+1] = ck
    Qki[i][i+1] = -sk
    Qki[i+1][i] = sk
    return Qki

#L?? arquivos input-a e input-b
def readfile_ab(file):
    A = np.loadtxt(file, dtype = 'float', delimiter = ' ',skiprows=1)
    # print(A)
    return A   
#L?? arquivo input-c
def readfile_c(file):
    dados = np.loadtxt(file, dtype = 'float', delimiter = ' ',max_rows=2)
    # print(dados)   
    barras = np.loadtxt(file, dtype = 'float', delimiter = ' ',skiprows=2)
    # print(barras)
    return dados, barras

def MatrizC():
    dados, barras = readfile_c('input-c')
    n = int(2*dados[0][1]) # n??mero de n??s n??o fixos
    rho = dados[1][0] # massa espec??fica do material [kg/m^3]
    A = dados[1][1] # ??rea [m^2]
    E = dados[1][2] # m??dulo de elasticidade [GPa]

    # Constru????o da matriz de massa
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

    # Constru????o da matriz de rigidez
    K = np.zeros([n,n])

    # Percorre as colunas da matriz que cont??m dos dados da barra
    for i,j,k,l in barras: # i e j - n??s; k = ??ngulo [??]; l = comprimento da barra [m]
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
    return A, M12, M,K
def calc_lambda_b():
    n=20
    list_lambda = []
    for i in range(n):
        lambda_i = 1/2*(1-np.cos((2*(i+1)-1)*np.pi/(2*n+1))**(-1))
        list_lambda.append(lambda_i)
    return list_lambda
def gif(h,n, escala = 1, index = 1):
    index = index - 1 # Define qual frequ??ncia ser?? usada, indo de 1 at?? 5, sendo 1 a frequ??ncia de menor valor
    # e 5 a de maior valor
    A,M12,M,K = MatrizC() 
    dados, barras = readfile_c('input-c')
    a,b,H = Householder(A)
    # print(a)
    c = np.zeros(len(a))
    c[:len(b)] = np.copy(b)
    V, Autovalores = QR_espectral(a,b,c,H)
    freq_5 = np.sort(Autovalores)[:5]
    V_novo = np.dot(M12,V)
    V_5 = np.zeros([5,len(V)])
    for i in range(len(freq_5)):
        V_5[i] = np.transpose(V_novo)[Autovalores==freq_5[i]]
    V_5 = np.transpose(V_5)
    freq_5 = np.sqrt(freq_5)
    frequencia_atual = freq_5[index]
    Z = V_5[:,index]
    for i in range(n):
        X = Z * np.cos(2*np.pi*frequencia_atual* i * h)

        pos_no = []
        X = escala * X
        #Posi????o dos n??s
        pos_no.append([5+X[0],40+X[1]]) 
        pos_no.append([-5+X[2],40+X[3]])
        pos_no.append([15+X[4],30+X[5]])
        pos_no.append([5+X[6],30+X[7]]) 
        pos_no.append([-5+X[8],30+X[9]]) 
        pos_no.append([-15+X[10],30+X[11]]) 
        pos_no.append([15+X[12],20+X[13]])
        pos_no.append([5+X[14],20+X[15]]) 
        pos_no.append([-5+X[16],20+X[17]]) 
        pos_no.append([-15+X[18],20+X[19]])
        pos_no.append([5+X[20],10+X[21]])
        pos_no.append([-5+X[22],10+X[23]]) 
        pos_no.append([10,0]) 
        pos_no.append([-10,0]) 
    
        for b in barras:
            no1 = int(b[0] - 1)
            no2 = int(b[1] - 1)
            plt.plot([ pos_no[no1][0] , pos_no[no2][0] ],[pos_no[no1][1],pos_no[no2][1]],'-b')
        plt.title('Tempo t = %f'%(i*h))
        plt.axis('equal')
        plt.xlim([-20, 20])
        plt.ylim([-10, 50])
        plt.figure(1)
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()
#Fun????o principal
def main():
    tarefa = input("Escolha a tarefa [1-Tarefa 4.1; 2-Tarefa 4.2]: ")
    while tarefa != '1' and tarefa != '2':
        print("Escolha uma op????o v??lida")
        tarefa = input("Escolha a tarefa [1-Tarefa 4.1; 2-Tarefa 4.2]: ")
    if tarefa == '1':
        item = input("Escolha o item da tarefa 4.1 [a-item (a); b-item(b)]: ")
        while item != 'a' and item != 'b':
            print("Escolha uma op????o v??lida")
            item = input("Escolha o item da tarefa 4.1 [a-item (a); b-item(b)]: ")
        if item == 'a':
            A = readfile_ab('input-a')
            analitico = '2'
        else:
            A = readfile_ab('input-b')
            analitico = input('Quer ver os autovalores anal??ticos [1-sim; 2-n??o]: ')
            while analitico != '1' and analitico != '2':
                print("Escolha uma op????o v??lida")
                analitico = input('Quer ver os autovalores anal??ticos [1-sim; 2-n??o]: ')
        printar = input('Quer ver os autovalores e autovetores obtidos [1-sim; 2-n??o]: ')
        while printar != '1' and printar != '2':
            print("Escolha uma op????o v??lida")
            printar = input('Quer ver os autovalores e autovetores obtidos [1-sim; 2-n??o]: ')
        verifica = input("Voc?? quer ver as verifica????es (Se A.v = lambda*v e se a matriz V ?? ortogonal) [1-sim; 2-n??o]: ")
        while verifica != '1' and verifica != '2':
            print("Escolha uma op????o v??lida")
            verifica = input("Voc?? quer ver as verifica????es (Se A*v = lambda*v e se a matriz V ?? ortogonal) [1-sim; 2-n??o]: ")
        
    else:
        A,M12,M,K = MatrizC() 
        printar = input('Voc?? quer ver as 5 menores frequ??ncias naturais do sistema [1-sim; 2-n??o]: ')
        while printar != '1' and printar != '2':
            print("Escolha uma op????o v??lida")
            printar = input('Voc?? quer ver as 5 menores frequ??ncias naturais do sistema [1-sim; 2-n??o]: ') 
        ter_gif = input('Voc?? quer ver uma anima????o da vibra????o [1-sim; 2-n??o]: ') 
        while printar != '1' and printar != '2':
            print("Escolha uma op????o v??lida")
            ter_gif = input('Voc?? quer ver uma anima????o da vibra????o [1-sim; 2-n??o: ') 
    a,b,H = Householder(A)
    # print(a)
    c = np.zeros(len(a))
    c[:len(b)] = np.copy(b)
    V, Autovalores = QR_espectral(a,b,c,H)
    if tarefa == '1':

        if printar == '1':
            print('Autovalores: ', Autovalores)
            print('Autovetores: ', V)
            if analitico == '1' and item == 'b':
                Autovalores_analitico = calc_lambda_b()
                print(Autovalores_analitico)
        if verifica == '1':
            V_t = np.transpose(V)
            V_aux = np.transpose(np.dot(A,V))
            print('\nVerifica????o se A.v = lambda*v')
            V_i = np.zeros([len(V),len(V)])
            for i in range(len(V_t)):
                V_i[i] = Autovalores[i]*V_t[i]
                print('Produto A.v:', V_aux[i])
                print('Produto lambda*v:', V_i[i])
            print('\nVerifica????o se a matriz V ?? ortogonal:\n Resultado de V^t.V: \n',np.dot(np.transpose(V),V))


    if tarefa == '2':
        freq = np.sqrt(Autovalores)
        freq_5 = np.sort(Autovalores)[:5]
        V_novo = np.dot(M12,V)
        # print(np.dot(np.transpose(V),V))  
        V_5 = np.zeros([5,len(V)])
        for i in range(len(freq_5)):
            V_5[i] = np.transpose(V_novo)[Autovalores==freq_5[i]]
            if printar == '1':
                print('Frequ??ncia %d: '%(i+1), np.sqrt(freq_5[i]))
                print('Modos de vibrar %d: '%(i+1), np.transpose(V_5[i]))
        if ter_gif == '1':
            h = float(input('Escolha o intervalo de tempo em segundos entre cada imagem: '))
            n = int(input('Escolha o n??mero de quadros'))
            escala1 = float(input('Defina a escala das deforma????es: '))
            index1 = int(input('Escolha a frequ??ncia [1-menor freq. at?? 5-maior freq.]: '))
            gif(h,n, escala= escala1, index=index1)
        
# gif(0.01,1000, escala = 1000, index = 1)
if __name__ == '__main__':
   main()