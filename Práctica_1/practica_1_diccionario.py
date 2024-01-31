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
frec_en_char, frec_esp_char = frec()[0], frec()[1]
"""
i) A partir de las muestras dadas, hallar el código Huffman binario de SEng y SEsp, y sus longitudes
medias L(SEng) y L(SEsp). Comprobar que se satisface el Primer Teorema de Shannon.
(1.50 puntos)
"""


def reordena_en():
    
    while len(frec_en) > 1:
       
        svalue_en = heapq.nsmallest(2, frec_en.values())
        skey_en = [clave for clave, valor in frec_en.items() if valor in svalue_en]
        k1_en, k2_en = skey_en[0], skey_en[1]
        
        print(f"k1_en = {k1_en}, k2_en = {k2_en}")
        print(f"Lo nuevo es {k1_en + k2_en}: {sum(svalue_en)}")
        frec_en.update({k1_en + k2_en: sum(svalue_en)})
        frec_en.pop(k1_en)
        frec_en.pop(k2_en)
        
        frec_en1 = dict(sorted(Counter(frec_en).items(), key=lambda item: item[1]))

    return frec_en1
        


def reordena_esp():
    
    while len(frec_esp) > 1:
    
        svalue_esp = heapq.nsmallest(2, frec_esp.values())
        skey_esp = [clave for clave, valor in frec_esp.items() if valor in svalue_esp]
        k1_esp, k2_esp = skey_esp[0], skey_esp[1]

        frec_esp.update({k1_esp + k2_esp: sum(svalue_esp)})
        frec_esp.pop(k1_esp)
        frec_esp.pop(k2_esp)

        frec_esp1 = dict(sorted(Counter(frec_esp).items(), key=lambda item: item[1]))
        
    return frec_esp1

















