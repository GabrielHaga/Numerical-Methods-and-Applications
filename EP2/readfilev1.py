import numpy as np
# def readfile_ab(file):
#     A = np.loadtxt(file, dtype = 'float', delimiter = ' ',skiprows=1)
#     # print(A)
#     return A
def readfile_c(file):
    dados = np.loadtxt(file, dtype = 'float', delimiter = ' ',max_rows=2)
    # print(dados)
    barras = np.loadtxt(file, dtype = 'float', delimiter = ' ',skiprows=2)
    # print(barras)
    return dados, barras
A = 

