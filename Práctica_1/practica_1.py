from collections import Counter
import heapq
import math
from math import sqrt as sqrt

print("---------------\nPRIMER APARTADO \n---------------")

with open("GCOM2024_pract1_auxiliar_eng.txt", 'r', encoding='utf-8') as f:
    en = f.read()
    
with open("GCOM2024_pract1_auxiliar_esp.txt", 'r', encoding='utf-8') as f:
    esp = f.read()
    
with open("lorentz.txt", 'r', encoding='utf-8') as f:
    lorentz = f.read()
    
def frec():

    # Filtrar caracteres
    caracter_en = [c for c in en]
    caracter_esp = [c for c in esp]

    # Contar la frecuencia de cada elemento
    frec_en = dict(sorted(Counter(caracter_en).items(), key=lambda item:item[1]))
    frec_esp = dict(sorted(Counter(caracter_esp).items(), key=lambda item:item[1]))

    return frec_en, frec_esp

frec_en, frec_esp = frec()[0], frec()[1]
frec_en_char, frec_esp_char = frec()[0], frec()[1]
k_esp, k_en = [], []
c_esp, c_en = [], []

k_esp, k_en = [], []
c_esp, c_en = [], []

def tree_nodes():
    """
    c_esp : list; dupla (nodo, {0,1}) de caracteres en español
    c_en : list; dupla (nodo, {0,1}) de caracteres en inglés
    """
    
    while len(frec_esp) > 1:
    
        # Los dos valores más pequeños y claves
        # con los valores más pequeños
        svalue_esp = heapq.nsmallest(2, frec_esp.values())
        skey_esp = [clave for clave, valor in frec_esp.items() if valor in svalue_esp]
        k1_esp, k2_esp = skey_esp[0], skey_esp[1]
        
        # Todos los nodos no terminales del árbol
        k_esp.append(k1_esp + k2_esp)
        
        # Codificado del arbol
        c_esp.append((k1_esp, 0))
        c_esp.append(((k2_esp, 1)))
        
        # Actualiza el diccionario
        frec_esp.update({k1_esp + k2_esp: sum(svalue_esp)})
        frec_esp.pop(k1_esp)
        frec_esp.pop(k2_esp)
        
        # Ordena el diccionario
        frec_esp1 = dict(sorted(Counter(frec_esp).items(), key=lambda item: item[1]))
        
    while len(frec_en) > 1:
    
        # Los dos valores más pequeños y claves con los valores más pequeños
        svalue_en = heapq.nsmallest(2, frec_en.values())
        skey_en = [clave for clave, valor in frec_en.items() if valor in svalue_en]
        k1_en, k2_en = skey_en[0], skey_en[1]
        
        # Todos los nodos no terminales del árbol
        k_en.append(k1_en + k2_en)
        
        # Codificado del arbol
        c_en.append((k1_en, 0))
        c_en.append(((k2_en, 1)))
        
        # Actualiza el diccionario
        frec_en.update({k1_en + k2_en: sum(svalue_en)})
        frec_en.pop(k1_en)
        frec_en.pop(k2_en)
        
        # Ordena el diccionario
        frec_en1 = dict(sorted(Counter(frec_en).items(), key=lambda item: item[1]))
        
    return c_esp, c_en

tree_esp, tree_en = tree_nodes()[0], tree_nodes()[1]

def encode_alph():
    """
    dictionary_esp : list; codificación Huffman de caracteres en español
    dictionary_en : list; codificación Huffman de caracteres en inglés
    """
    # Listas de claves
    chars_esp = [x for x in frec_esp_char.keys()]
    chars_en = [x for x in frec_en_char.keys()]
    # Listas de nodos
    nodes_esp = [node[0] for node in tree_esp]
    nodes_en = [node[0] for node in tree_en]

    dictionary_esp, dictionary_en = [], []
    
    for char in chars_esp:
        
        huffman_binary = ""
        
        for i in range(len(tree_esp)):
            if char in nodes_esp[i]:
                huffman_binary = str(tree_esp[i][1]) + huffman_binary
                
        dictionary_esp.append((char, huffman_binary))
        
    for char in chars_en:
        huffman_binary = ""
        
        for i in range(len(tree_en)):
            if char in nodes_en[i]:
                huffman_binary = str(tree_en[i][1]) + huffman_binary
                
        dictionary_en.append((char, huffman_binary))

    return dictionary_esp, dictionary_en

dict_esp, dict_en = encode_alph()[0], encode_alph()[1]
print(f"Código Huffman de S_eng: \n{dict_en}\n\n\
      Código Huffman de S_esp:\n{dict_esp}\n")

# Listas de código Huffman de cada caracter
huffman_en, huffman_esp = [],[]
for duple in dict_en:
    huffman_en.append(duple[1])
    
for duple in dict_esp:
    huffman_esp.append(duple[1])
    
def probabilidades():
    """
    probabilidad_en : list; frecuencias relativas
                      de caracteres en inglés
    probabilidad_esp : list; frecuencias relativas
                      de caracteres en español
    """
    
    N_en, N_esp = len(en), len(esp)
    probabilidad_en = list(map(lambda x: x/N_en, list((frec_en_char.values()))))
    probabilidad_esp = list(map(lambda x: x/N_esp, list((frec_esp_char.values()))))

    return probabilidad_en, probabilidad_esp

W_en, W_esp = probabilidades()[0], probabilidades()[1]

def longitud_media():
    """
    L_en : float; longitud media del código Huffman en inglés
    L_esp : float; longitud media del código Huffman en español
    """
    L_en, L_esp = 0, 0 
    
    for i in range (len(W_en)):
        L_en += W_en[i] * len(dict_en[i][1])
        
    for i in range (len(W_esp)):
        L_esp += W_esp[i] * len(dict_esp[i][1])
        
    return L_en, L_esp

def error_longitud():
    """
    error_en : float; error de longitud en inglés
    error_esp : float; error de longitud en español
    """
    
    e_en, e_esp = 0, 0 
    
    for i in huffman_en:
        e_en += (abs(len(i))**2)
        
    
    for i in huffman_esp:
        e_esp += (abs(len(i))**2)
        
    error_en = (1/len(en))*sqrt(e_en)
    error_esp = (1/len(esp))*sqrt(e_esp)
    
    return error_en, error_esp
        
def error_entropia():
    """
    error_en : float; error de entropía en inglés
    error_esp : float; error de entropía en español
    """
    e_en, e_esp = 0, 0
    
    for i in range(len(frec_en_char)):
        e_en += (abs(math.log2(W_en[i]) + 1/math.log(2)))**2
        
    for i in range(len(frec_esp_char)):
        e_esp += (abs(math.log2(W_esp[i]) + 1/math.log(2)))**2
        
    error_en = (1/len(en))*sqrt(e_en)
    error_esp = (1/len(esp))*sqrt(e_esp)
    
    return error_en, error_esp

error_en_long, error_esp_long = error_longitud()[0], error_longitud()[1]
error_en, error_esp = error_entropia()[0], error_entropia()[1]
L_en, L_esp = longitud_media()[0], longitud_media()[1]

print(f"Las longitudes medias son: \nL_en = {round(L_en, 3)} con error de\
 entropía {round(error_en, 3)} y error de longitud {round(error_en_long,3)} \
      \nL_esp = {round(L_esp, 3)} con error de entropía {round(error_esp, 3)}\
 y error de longitud {round(error_esp_long,3)}")

def entropia():
    """
    -entropia_en : float; entropía total del sistema, caso inglés
    -entropia_esp : float; entropía total del sistema, caso español
    """
    entropia_en, entropia_esp = 0, 0 
    
    for i in range(len(W_en)):
        entropia_en += W_en[i]*(math.log2(W_en[i]))
        
    for i in range(len(W_esp)):
        entropia_esp += W_esp[i]*(math.log2(W_esp[i]))
        
    return -entropia_en, -entropia_esp
        
entropia_en, entropia_esp = entropia()[0], entropia()[1]
         
def shannon_th(L_en, L_esp):
    """
    L_en : float; longitud media del código Huffman en inglés
    L_esp : float; longitud media del código Huffman en español
    check_shannon_en : str; print si cumple el teorema de Shannon, caso inglés
    check_shannon_esp : str; print si cumple el teorema de Shannon, caso español
    """
    
    if entropia_en <= L_en and L_en <= entropia_en + 1:
        check_shannon_en = "S_en cumple el teorema de Shannon"   
    else: check_shannon_en = "S_en no cumple el teorema de Shannon"
        
    if entropia_esp <= L_esp and L_esp <= entropia_esp + 1:
        check_shannon_esp = "S_esp cumple el teorema de Shannon"
    else: check_shannon_esp = "S_esp no cumple el teorema de Shannon"
        
    return check_shannon_en, check_shannon_esp

shannon_en, shannon_esp = shannon_th(L_en, L_esp)[0], shannon_th(L_en, L_esp)[1]
print(f"\nTeorema de Shannon: Entropía <= Longitud media <= Entropía + 1\
      \nInglés: {entropia_en} <= {L_en} <= {entropia_en + 1}\
    \n{shannon_en}\
      \nEspañol: {entropia_esp} <= {L_esp} <= {entropia_esp + 1}\
     \n{shannon_esp}")

print("\n----------------\nSEGUNDO APARTADO \n----------------")

def encode():
    """
    en_encode : str; código Huffman de texto inglés
    esp_encode : str; código Huffman de texto español
    lorentz_en : str; código Huffman de Lorentz en inglés
    lorentz_esp : str; código Huffman de Lorentz en español
    """
    en_encode  = ""
    for char in en:
        for i in range(len(dict_en)):
            if char == dict_en[i][0]:
                en_encode += dict_en[i][1]
            
    esp_encode  = ""
    for char in esp:
        for i in range(len(dict_esp)):
            if char == dict_esp[i][0]:
                esp_encode += dict_esp[i][1]
                

    lorentz_en, lorentz_esp = "", ""
    for char in lorentz:
        for i in range(len(dict_en)):
            if char == dict_en[i][0]:
                lorentz_en += dict_en[i][1]
                
    for char in lorentz:
        for i in range(len(dict_esp)):
            if char == dict_esp[i][0]:
                lorentz_esp += dict_esp[i][1]
                
    return en_encode, esp_encode, lorentz_en, lorentz_esp

en_encode, esp_encode = encode()[0], encode()[1]
lorentz_en, lorentz_esp = encode()[2], encode()[3]

print(f"Lorentz en la codificación en inglés: {lorentz_en}, \
con longitud {len(lorentz_en)}")
print(f"Lorentz en la codificación en español: {lorentz_esp}, \
con longitud {len(lorentz_esp)}")

def binary_encode():
    """
    binary_lorentz : str; codificación binaria de Lorentz
    """
    binary_lorentz = ""
    for letter in lorentz:
        b = bin(ord(letter))[2:]
        binary_lorentz += b
    
    return binary_lorentz

binary_lorentz = binary_encode()

print(f"\nLorentz en la codificación binaria: \n{binary_lorentz}, con longitud {len(binary_lorentz)} \
      \nLa codificación binaria es {len(binary_lorentz)/len(lorentz_en)} más larga \
respecto a la codificación inglesa\
\nLa codificación binaria es {len(binary_lorentz)/len(lorentz_esp)} más larga respecto a \
la codificación española")


print("\n---------------\nTERCER APARTADO \n---------------")

# Listas de código Huffman de cada caracter
huffman_en, huffman_esp = [],[]
for duple in dict_en:
    huffman_en.append(duple[1])
    
for duple in dict_esp:
    huffman_esp.append(duple[1])
    
def decode(word, lenguage):
    """
    word : str; palabra en código Huffman a decodificar
    lenguage : str; {eng, esp}, lenguaje seleccionado para decodificar
    decode_en : str; decodificado de word en inglés
    decode_esp : str; decodificado de word en español
    """
    
    if lenguage != "eng" and lenguage != "esp":
        print("El lenguaje debe ser eng o esp")

    else:
        
        decode_en, decode_esp = "", ""

        if lenguage == "eng":

            i = 0
            while len(word) >= 3:

                length = len(huffman_en[i])
                if huffman_en[i] in word[:length]:
                    decode_en += dict_en[i][0]
                    word = word[length:]
                    i = 0
                i += 1

            return decode_en

        if lenguage == "esp":
            
            i = 0
            while len(word) >= 3:
                
                length = len(huffman_esp[i])
                if huffman_esp[i] in word[:length]:
                    decode_esp += dict_esp[i][0]
                    word = word[length:]
                    i = 0
                i += 1

            return decode_esp
        
print("Tomando las codificaciones de Lorentz del apartado anterior:\n")
print("Decodificado de 10011100011001000001011101011110100 en inglés:",
      decode("10011100011001000001011101011110100", "eng"))
print("Decodificado de 11011111101101100101010111011101101 en español:",
      decode("11011111101101100101010111011101101", "esp"))
