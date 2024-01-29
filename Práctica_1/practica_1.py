from collections import Counter


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
    
    for l in frec_esp.keys():
        letras.append(l)
        
    for l in frec_en.keys():
        letters.append(l)
        
    for k in range (0,len(letters), 2):
        print("new_key =", letters[k]+letters[k+1], "; new_value =", frec_en[letters[k]] + frec_en[letters[k+1]] )
        new_l = {letters[k]+letters[k+1] : frec_en[letters[k]] + frec_en[letters[k+1]]}
        
        new1_l =  {letters[k]:frec_en[letters[k]]}
        new2_l =  {letters[k+1]:frec_en[letters[k+1]]}
        
        print("new_l =", new_l)
        print("new1_l =", new1_l)
        print("new2_l =", new2_l)

        print("frecn_en.pop =", frec_en.pop(letters[k]))
        print("frecn_en.pop =", frec_en.pop(letters[k+1]))
        print("frec_en.update(new_l) = ", frec_en.update(new_l))
        frec_en1 = dict(sorted(Counter(frec_en).items(), key=lambda item:item[1]))
        
        print("frec_en = ", frec_en1)
        print("\n")
        for l in frec_en.keys():
            letters.append(l)
            
        

    
    
    
    
    
    
    """
    while len(letters) != 1:
        print("Antes", letters,"\n")
        raiz.append(letters[0]+letters[1])
        letters.pop(0)
        letters.pop(0)
        if len(letters) == 1:
            raiz.append(letters[0])
    """      
    return raiz



















