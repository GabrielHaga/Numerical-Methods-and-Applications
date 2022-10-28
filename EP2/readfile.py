import numpy as np
a = open("input-c",'r+').read()
# c = np.loadtxt(a)
l_a = []
aux = []
num = ''
# print(float('9.1'))
# char = 'a'
# char1 = 'b'
# print(char+char1)
print(a[9:17])
# for i in range(len(a)):
#     print(i)
#     if a[i] == ' ' or a[i] == '\n':
#         aux.append(float(num))
#         num = ''
        
#         # print('a')
#     # elif a[i] != '.':

#         if a[i] == '\n':
#             l_a.append(aux)
#             aux = []
#             # num = ''
#         # print('sim')
#     else:
#         num = num+a[i]
#         print()
        # print(num)
# l_a = np.array(l_a)
print(l_a)
# list_of_lists = []
# with open('input-c') as f:
#     for line in f:
#         inner_list = [elt.strip() for elt in line.split(',')]
#         # in alternative, if you need to use the file content as numbers
#         # inner_list = [int(elt.strip()) for elt in line.split(',')]
#         list_of_lists.append(inner_list)

# print(list_of_lists)
def matriz(file):
    matriz= np.loadtxt(file, dtype = 'float', delimiter = ' ',skiprows=1)
    #print(matriz)
    return matriz