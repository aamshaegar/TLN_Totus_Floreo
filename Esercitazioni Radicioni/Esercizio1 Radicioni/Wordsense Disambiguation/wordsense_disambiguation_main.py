import semcore
import sys

__name__ = "__main__"


def main():

    if len(sys.argv) < 4:
        print("\n***************************************")
        print("Errore: Parametri non corretti!\nESEMPIO: python wordsense_disambiguation_main.py N_SENT MAX_LEMMAS N_ITERATION")
        print("dove, N_SENT = numero di frasi da testare")
        print("dove, MAX_LEMMAS = massimo numero di sostantivi per blocco di N_SENT frasi scelte casualmente")
        print("dove, N_ITERATION = numero di test da eseguire. Ad ogni test vengono selezionate in modo casuale N_SENT frasi")
        print("***************************************")
        exit()

    accuracy = semcore.testing_WSD_on_semcor(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    print("Accuratezza totale: " + str(accuracy) + "%")


if __name__ == "__main__":
    main()

