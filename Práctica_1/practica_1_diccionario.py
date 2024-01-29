from collections import Counter
import heapq

"""
Sean SEng = {a, b, c, ...,Z, !, ?,−, /n, , 0, ..,9} y SEsp = {a,á, b, c, ...,ñ, ...,Z,¡, !,¿, ?,−, /n, , 0, ..,9}
los alfabetos del inglés y el español, respectivamente. Consideramos una pequeña muestra de
cada idioma en los siguientes archivos:
• “GCOM2024_pract1_auxiliar_eng.txt” con la muestra en inglés.
• “GCOM2024_pract1_auxiliar_esp.txt” con la muestra en español.
Suponemos que las mayúsculas y minúsculas son distintos estados de las variables. Del mismo
modo, las vocales con y sin tilde también se consideran estados distintos, así como los espacios
y otros signos.
"""
def frec():
    with open("GCOM2024_pract1_auxiliar_eng.txt", 'r', encoding='utf-8') as f:
        en = f.read()
        
    with open("GCOM2024_pract1_auxiliar_esp.txt", 'r', encoding='utf-8') as f:
        esp = f.read()

    # Filtrar caracteres
    caracter_en = [c for c in en]
    caracter_esp = [c for c in esp]

    # Contar la frecuencia de cada elemento
    frec_en = dict(sorted(Counter(caracter_en).items(), key=lambda item:item[1]))
    frec_esp = dict(sorted(Counter(caracter_esp).items(), key=lambda item:item[1]))

    return frec_en, frec_esp


frec_en, frec_esp = frec()[0], frec()[1]

"""
i) A partir de las muestras dadas, hallar el código Huffman binario de SEng y SEsp, y sus longitudes
medias L(SEng) y L(SEsp). Comprobar que se satisface el Primer Teorema de Shannon.
(1.50 puntos)
"""

letters, letras, raiz = [], [], []
#raiz = {}

def reordena():
    
    while len(frec_en) > 1:
        """
        print("frecn_en.pop =", frec_en.pop(letters[k]))
        print("frecn_en.pop =", frec_en.pop(letters[k+1]))
        print("frec_en.update(new_l) = ", frec_en.update(new_l))
        frec_en1 = dict(sorted(Counter(frec_en).items(), key=lambda item:item[1]))
        
        print("frec_en = ", frec_en1)
        print("\n")
        """
        svalue = heapq.nsmallest(2, frec_en.values())
        skey = [clave for clave, valor in frec_en.items() if valor in svalue]
        k1, k2 = skey[0], skey[1]
        
        frec_en.update({k1 + k2: sum(svalue)})
        frec_en.pop(k1)
        frec_en.pop(k2)
        
        frec_en1 = dict(sorted(Counter(frec_en).items(), key=lambda item: item[1]))
        
        """
        min(frec_en, key=frec_en.get)
        list(frec_en.keys())[2"]
        """
    return frec_en1
        

















