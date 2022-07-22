import sys
from false_friends import *

__name__ = "__main__"


def main():

    if len(sys.argv) < 4:
        print("\n***************************************")
        print("Errore: Parametri non corretti!\nEsempio: python main.py SENT EDIT_THRESHOLD SIM_THRESHOLD")
        print("dove, SENT = numero di frasi da prelevare dal corpus Semcor. Max 20 frasi")
        print("dove, EDIT_THRESHOLD = valore soglia limite per la distanza di edit")
        print("dove, SIM_THRESHOLD = valore soglia limite per la similarità tra termini")
        print("\nValori consigliati: python main.py 20  2  0.3")
        print("***************************************\n")
        exit()


    number_sentence = int(sys.argv[1])
    if number_sentence > 20:
        number_sentence = 20

    edit_threshold = int(sys.argv[2])
    similarity_threshold = float(sys.argv[3])

    couple_terms = random_couples(number_sentence, edit_threshold, similarity_threshold)
    print()
    for el in couple_terms:
        print(el)
    print()



if __name__ == "__main__":
    main()