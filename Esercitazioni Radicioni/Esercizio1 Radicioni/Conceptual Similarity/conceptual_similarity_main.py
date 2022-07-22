from conceptualSimilarity import conceptualSimilarity
import sys

__name__ = "__main__"


def main():

    if len(sys.argv) < 4:
        print("\n***************************************")
        print("Errore: Parametri non corretti!\nESEMPIO: python conceptual_similarity_main.py csv_path_file SIMILARITY CORRELATION")
        print("dove, SIMILARITY in [wupalmer, shortest, leacock]")
        print("dove, CORRELATION in [pearson, spearman]")
        print("***************************************")
        exit()

    """
        -----------------------------------------
        SIMILARITY METRIC: leakcock_chodorow
        -----------------------------------------

        Words:              LOVE - SEX
        Metric similarity:  2.327277705584417
        Human similarity:   6.77
    """

    conceptual = conceptualSimilarity(sys.argv[1])
    conceptual.calculate_similarity(sys.argv[2])
    correlation = conceptual.similarity_correlation(sys.argv[3])
    print("-----------------------------------------")
    print("METRICA DI SIMILARITA': ", sys.argv[2])
    print("-----------------------------------------")
    for el in conceptual.get_similarity():
        print("\nParole: ", el[0] + " - " + el[1])
        print("Similarità calcolata: ", el[2])
        print("Similarità wordSin353: ", el[3])
    
    print("\nCorrelazione di " + sys.argv[3] + " : " + str(correlation))

if __name__ == "__main__":
    main()

