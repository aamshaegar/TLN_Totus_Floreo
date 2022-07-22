import time
from NER import NERtagger

__name__ = "__main__"

def main():

    
    start = time.time()
    ner = NERtagger("training_set_ITA.txt")
    end = time.time()
    print("secondi per la lettura del file: ",end - start)

    print("\nTagset")
    tagset = ner.get_tagSet() 
    print(tagset)
    
    """
    print()
    start = time.time()
    print("Inizio decoding")
    smoothing = "smoothing1"
    print("smoothing: ", smoothing)
    ner.testing_viterbi("test_set_ITA.txt", smoothing)
    end = time.time()
    metrics, accuracy = ner.get_evaluating_metrics()
    for tag in metrics:
        print(tag, metrics[tag])
    print()
    for tag in accuracy:
        print(tag, accuracy[tag])
    end = time.time()
    print("secondi per viterbi: ", end - start)
        

    smoothing = "smoothing2"
    print("smoothing: ", smoothing)
    ner.testing_viterbi("test_set_ITA.txt", smoothing)
    end = time.time()
    metrics, accuracy = ner.get_evaluating_metrics()
    for tag in metrics:
        print(tag, metrics[tag])
    print()
    for tag in accuracy:
        print(tag, accuracy[tag])
    end = time.time()
    print("secondi per viterbi: ", end - start)
    """

    smoothing = "smoothing3"
    print("smoothing: ", smoothing)
    ner.testing_viterbi("test_set_ITA.txt", smoothing)
    end = time.time()
    metrics, accuracy = ner.get_evaluating_metrics()
    for tag in metrics:
        print(tag, metrics[tag])
    print()
    for tag in accuracy:
        print(tag, accuracy[tag])
    end = time.time()
    print("secondi per viterbi: ", end - start)

    """
    smoothing = "validation_set_ENG.txt"
    print("smoothing: ", smoothing)
    ner.testing_viterbi("test_set_ITA.txt", smoothing)
    end = time.time()
    metrics, accuracy = ner.get_evaluating_metrics()
    for tag in metrics:
        print(tag, metrics[tag])
    print()
    for tag in accuracy:
        print(tag, accuracy[tag])
    end = time.time()
    print("secondi per viterbi: ", end - start)
    """


    # Risultato del testing dell'algoritmo di viterbi su "test_set_ITA"
    # il test set analizzato contiene 11069 frasi
    #
    # testing su test_set_ita (tutto) con smoothing1:
    # accuratezza:  96.637%
    #   ORG {'COR': 1729, 'PAR': 0, 'MIS': 500, 'HYP': 326, 'PRECISION': 84.136%, 'RECALL': 77.568%}
    #   PER {'COR': 5886, 'PAR': 174, 'MIS': 2491, 'HYP': 1512, 'PRECISION': 77.734%, 'RECALL': 70.264%}
    #   MISC {'COR': 1196, 'PAR': 343, 'MIS': 1146, 'HYP': 637, 'PRECISION': 54.963%, 'RECALL': 51.067%}
    #   LOC {'COR': 7738, 'PAR': 62, 'MIS': 2050, 'HYP': 1196, 'PRECISION': 86.016%, 'RECALL': 79.056%}
    # 
    # testing su test_set_ita (tutto) con smoothing2:
    # accuratezza:  96.657%
    #   ORG {'COR': 1730, 'PAR': 0, 'MIS': 499, 'HYP': 322, 'PRECISION': 84.308%, 'RECALL': 77.613%}
    #   PER {'COR': 5896, 'PAR': 221, 'MIS': 2481, 'HYP': 1512, 'PRECISION': 77.284%, 'RECALL': 70.383%}
    #   MISC {'COR': 1215, 'PAR': 250, 'MIS': 1127, 'HYP': 761, 'PRECISION': 54.582%, 'RECALL': 51.879%}
    #   LOC {'COR': 7740, 'PAR': 62, 'MIS': 2048, 'HYP': 1200, 'PRECISION': 85.981%, 'RECALL': 79.076%}
    # 
    # testing su test_set_ita (tutto) con smoothing3: 
    # accuratezza:  97.09%
    #   ORG {'COR': 1750, 'PAR': 0, 'MIS': 479, 'HYP': 319, 'PRECISION': 84.582%, 'RECALL': 78.511%}
    #   PER {'COR': 6808, 'PAR': 152, 'MIS': 1569, 'HYP': 718, 'PRECISION': 88.669%, 'RECALL': 81.27%}
    #   MISC {'COR': 1247, 'PAR': 257, 'MIS': 1095, 'HYP': 619, 'PRECISION': 58.738%, 'RECALL': 53.245%}
    #   LOC {'COR': 7753, 'PAR': 56, 'MIS': 2035, 'HYP': 1202, 'PRECISION': 86.039%, 'RECALL': 79.209%}
    # 
    # testing su test_set_ita (tutto) con smoothing4: 
    # accuratezza:  96.729%
    #   ORG {'COR': 1744, 'PAR': 0, 'MIS': 485, 'HYP': 325, 'PRECISION': 84.292%, 'RECALL': 78.241%}
    #   PER {'COR': 5952, 'PAR': 152, 'MIS': 2425, 'HYP': 1525, 'PRECISION': 78.018%, 'RECALL': 71.052%}
    #   MISC {'COR': 1217, 'PAR': 263, 'MIS': 1125, 'HYP': 640, 'PRECISION': 57.406%, 'RECALL': 51.964%}
    #   LOC {'COR': 7754, 'PAR': 56, 'MIS': 2034, 'HYP': 1211, 'PRECISION': 85.955%, 'RECALL': 79.219%}
    # 
    # provato su pc: intel i5 di quinta generazione con GPU: Nvidia 940M
    # tempo di esecuzione: circa 44 secondi per ogni test

    """
    print()
    print("Baseline\n")
    start = time.time()
    smoothing = "B-MISC"
    print("smoothing tag: ", smoothing)
    ner.testing_baseline("test_set_ITA.txt", smoothing)
    metrics, accuracy = ner.get_evaluating_metrics()
    for tag in metrics:
        print(tag, metrics[tag])
    print()
    for tag in accuracy:
        print(tag, accuracy[tag])
    end = time.time()
    print("secondi per baseline: ", end - start)

    
    print()
    print("Baseline\n")
    start = time.time()
    smoothing = "O"
    print("smoothing tag: ", smoothing)
    ner.testing_baseline("test_set_ITA.txt", smoothing)
    metrics, accuracy = ner.get_evaluating_metrics()
    print("accuratezza: ", accuracy)
    for tag in metrics:
        print(tag, metrics[tag])
    end = time.time()
    print("secondi per baseline: ", end - start)
    """
    
    # testing su test_set_ita_piccolo (tutto) con smoothing1: accuratezza = 85.642% | precisione = 66.369% | Recall = 76.667%
    #
    # risultato del testing della baselise semplice su "test_set_ITA"
    # smoothing1: se la parola è sconosciuta etichettala come O
    # smoothing2: se la parola è sconosciuta etichettala come B-MISC
    # il test set analizzato contiene 11069 frasi
    # 
    # testing su test_set_ita (tutto) con smoothing1: 
    #   accuratezza:  0.9524
    #   ORG {'COR': 1361, 'PAR': 443, 'MIS': 868, 'HYP': 542, 'PRECISION': 0.5801, 'RECALL': 0.6106}
    #   PER {'COR': 5011, 'PAR': 1552, 'MIS': 3366, 'HYP': 2656, 'PRECISION': 0.5436, 'RECALL': 0.5982}        
    #   MISC {'COR': 661, 'PAR': 1207, 'MIS': 1681, 'HYP': 849, 'PRECISION': 0.2433, 'RECALL': 0.2822}
    #   LOC {'COR': 7045, 'PAR': 805, 'MIS': 2743, 'HYP': 2286, 'PRECISION': 0.695, 'RECALL': 0.7198}
    # 
    # testing su test_set_ita (tutto) con smoothing2:
    #   accuratezza:  0.938
    #   ORG {'COR': 1361, 'PAR': 443, 'MIS': 868, 'HYP': 542, 'PRECISION': 0.5801, 'RECALL': 0.6106}
    #   PER {'COR': 5011, 'PAR': 1552, 'MIS': 3366, 'HYP': 2656, 'PRECISION': 0.5436, 'RECALL': 0.5982}        
    #   MISC {'COR': 897, 'PAR': 1115, 'MIS': 1445, 'HYP': 10192, 'PRECISION': 0.0735, 'RECALL': 0.383}        
    #   LOC {'COR': 7045, 'PAR': 805, 'MIS': 2743, 'HYP': 2286, 'PRECISION': 0.695, 'RECALL': 0.7198}

    # provato su pc: intel i5 di quinta generazione con GPU: Nvidia 940M
    # tempo di esecuzione: circa 5 secondi



    # testing su frasi Harry Potter
    """
    smoothing:  smoothing1
        [('La', 'O'), ('vera', 'O'), ('casa', 'O'), ('di', 'O'), ('Harry', 'B-MISC'), ('Potter', 'I-MISC'), ('è', 'O'), ('il', 'O'), ('castello', 'O'), ('di', 'O'), ('Hogwards', 'O')]
        [('Harry', 'B-PER'), ('le', 'O'), ('raccontò', 'O'), ('del', 'O'), ('loro', 'O'), ('incontro', 'O'), ('a', 'O'), ('Diagon', 'O'), ('Alley', 'O')]
        [('Mr', 'B-MISC'), ('Dursley', 'O'), ('era', 'O'), ('direttore', 'O'), ('di', 'O'), ('una', 'O'), ('ditta', 'O'), ('di', 'O'), ('nome', 'O'), ('Grunnings', 'O'), ('che', 'O'), ('fabbricava', 'O'), ('trapani', 'O')]

    smoothing:  smoothing2
        [('La', 'O'), ('vera', 'O'), ('casa', 'O'), ('di', 'O'), ('Harry', 'B-MISC'), ('Potter', 'I-MISC'), ('è', 'O'), ('il', 'O'), ('castello', 'O'), ('di', 'O'), ('Hogwards', 'O')]
        [('Harry', 'B-PER'), ('le', 'O'), ('raccontò', 'O'), ('del', 'O'), ('loro', 'O'), ('incontro', 'O'), ('a', 'O'), ('Diagon', 'B-MISC'), ('Alley', 'I-PER')]
        [('Mr', 'B-MISC'), ('Dursley', 'O'), ('era', 'O'), ('direttore', 'O'), ('di', 'O'), ('una', 'O'), ('ditta', 'O'), ('di', 'O'), ('nome', 'O'), ('Grunnings', 'O'), ('che', 'O'), ('fabbricava', 'O'), ('trapani', 'O')]

    smoothing:  smoothing3
        [('La', 'O'), ('vera', 'O'), ('casa', 'O'), ('di', 'O'), ('Harry', 'B-MISC'), ('Potter', 'I-MISC'), ('è', 'O'), ('il', 'O'), ('castello', 'O'), ('di', 'O'), ('Hogwards', 'O')]
        [('Harry', 'B-PER'), ('le', 'O'), ('raccontò', 'O'), ('del', 'O'), ('loro', 'O'), ('incontro', 'O'), ('a', 'O'), ('Diagon', 'B-PER'), ('Alley', 'I-PER')]
        [('Mr', 'B-MISC'), ('Dursley', 'I-MISC'), ('era', 'O'), ('direttore', 'O'), ('di', 'O'), ('una', 'O'), ('ditta', 'O'), ('di', 'O'), ('nome', 'O'), ('Grunnings', 'O'), ('che', 'O'), ('fabbricava', 'O'), ('trapani', 'O')]

    smoothing:  validation_set_ITA.txt
        [('La', 'O'), ('vera', 'O'), ('casa', 'O'), ('di', 'O'), ('Harry', 'B-MISC'), ('Potter', 'I-MISC'), ('è', 'O'), ('il', 'O'), ('castello', 'O'), ('di', 'O'), ('Hogwards', 'O')]
        [('Harry', 'B-PER'), ('le', 'O'), ('raccontò', 'O'), ('del', 'O'), ('loro', 'O'), ('incontro', 'O'), ('a', 'O'), ('Diagon', 'B-PER'), ('Alley', 'I-PER')]
        [('Mr', 'B-MISC'), ('Dursley', 'O'), ('era', 'O'), ('direttore', 'O'), ('di', 'O'), ('una', 'O'), ('ditta', 'O'), ('di', 'O'), ('nome', 'O'), ('Grunnings', 'O'), ('che', 'O'), ('fabbricava', 'O'), ('trapani', 'O')]
    """

    """

    sentences = ["La vera casa di Harry Potter è il castello di Hogwards", "Harry le raccontò del loro incontro a Diagon Alley", "Mr Dursley era direttore di una ditta di nome Grunnings che fabbricava trapani"]
    
    print()
    print("Prova baseline su harry Potter\n")
    smoothing = "O"
    print("smoothing: ", smoothing)
    for sent in sentences:
        best_path = ner.baseline(sent.split(), smoothing)
        print([(sent.split()[i],best_path[i]) for i in range(len(best_path))])
    

    print()
    print("Prova baseline su harry Potter\n")
    smoothing = "B-MISC"
    print("smoothing: ", smoothing)
    for sent in sentences:
        best_path = ner.baseline(sent.split(), smoothing)
        print([(sent.split()[i],best_path[i]) for i in range(len(best_path))])
    """

    """
    print()
    print("Prova viterbi su harry Potter\n")
    smoothing = "smoothing1"
    print("smoothing: ", smoothing)
    for sent in sentences:
        best_path = ner.viterbi(sent.split(), smoothing)
        print([(sent.split()[i],best_path[i]) for i in range(len(best_path))])

    print()
    smoothing = "smoothing2"
    print("smoothing: ", smoothing)
    for sent in sentences:
        best_path = ner.viterbi(sent.split(), smoothing)
        print([(sent.split()[i],best_path[i]) for i in range(len(best_path))])

    print()
    smoothing = "smoothing3"
    print("smoothing: ", smoothing)
    for sent in sentences:
        best_path = ner.viterbi(sent.split(), smoothing)
        print([(sent.split()[i],best_path[i]) for i in range(len(best_path))])

    print()
    smoothing = "validation_set_ITA.txt"
    print("smoothing: ", smoothing)
    for sent in sentences:
        best_path = ner.viterbi(sent.split(), smoothing)
        print([(sent.split()[i],best_path[i]) for i in range(len(best_path))])
    """

if __name__ == "__main__":
    main()

