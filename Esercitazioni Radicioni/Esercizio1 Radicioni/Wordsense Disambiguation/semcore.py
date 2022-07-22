from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
import numpy as np
import re
import random
import LeskAlgorithm

MAX_SENTENCES = 10000
#MAX_SENTENCES = len(semcor.sents())
#sentences, sentence_list = open_semcor("brown1/tagfiles/br-b20.xml", 50)


def open_semcor(xml_semcor_path, sentences_number):
    """
        Restituisce le prime "sentences_number" frasi tratte da un corpus 
        annotato di Semcor passato per parametro.
        xml_semcor path è il path al file xml corrispondente ad un sottoinsieme
        di frasi annotate tratte dal brown corpus
    """
    sentences = semcor._items(xml_semcor_path, "token", True, True, True)[:sentences_number]
    sentence_list = semcor.sents(xml_semcor_path)[:sentences_number]
    return sentences, sentence_list



def open_random_semcor(sentences_number):
    """
        Restituisce le prime "sentences_number" frasi scelte in modo casuale 
        da un insieme di corpus semcor annotati.
        La scelta avviene in un insieme di 37176 frasi, presenti nel corpus semcor
    """

    first_index = 0
    last_index = (MAX_SENTENCES - sentences_number)
    random_index = random.randint(first_index, last_index)
    last_element = random_index + sentences_number
    sentences = semcor._items(None, "token", True, True, True)[random_index:last_element]
    sentence_list = semcor.sents()[random_index:last_element]
    
    return sentences, sentence_list



def read_nouns_from_semcor(tagged_sentences):
    """
        Legge da una lista di frasi tratte da un corpus semcor
        i lemmi associati ai sostantivi (tag NN), quindi restituisce 
        una lista di terne formate da:
            - lemma: il lemma associato al sostantivo
            - synset: l'indicazione del synset associato al lemma, preso dal corpus annotato
            - index: indice della frase da cui è stato estratto il lemma
    """

    lemmas_list = []
    for i in range(len(tagged_sentences)):
        for couple in tagged_sentences[i]:
            if (len(couple) == 6) and (couple[1] == "NN") and (couple[4] != None) and (not ";" in couple[4]):
                _, _, lemma,_, synset, _ = couple
                lemmas_list.append([lemma,synset, i])
   
    return lemmas_list



def word_sense_disambiguation(lemmas_list, sentence_list):
    """
        Dati in input un insieme di lemmi di sostantivi tratti dal corpus semcor
        e un insieme di frasi da cui sono ricavati i lemmi, effettua la 
        disambiguazione del lemma nel contesto in cui occorre.
        La funzione invoca l'algoritmo di Lesk per ogni coppia (lemma,frase), 
        quindi calcola l'accuratezza totale sull'insieme di frasi.
    """

    if len(lemmas_list) == 0:
        print("Nessun lemma NN etichettato con l'inicazione del wordnetsynset è stato trovato nel sottoinsieme di frasi del corpus semcor. Nessuna parola da disambiguare!")
        return 0

    accuracy = 0
    for lemma in lemmas_list:
        lemma_synsets = wn.synsets(lemma[0])
        index = int(lemma[1])

        # gestione di synset non aggiornati nel corpus semcor, 
        # alcuni indici non corrispondono ai synset associati ai lemmi
        if (index == 0) or ((index + 1) > len(lemma_synsets)):
            continue
        
        best_sense = LeskAlgorithm.lesk(lemma[0], " ".join(sentence_list[lemma[2]]))
        lemma_synset = lemma_synsets[index - 1]
        #print("corretto: ", lemma_synset, "restituito: ", best_sense, "confronto: ", lemma_synset == best_sense)
        if str(lemma_synset) == str(best_sense):
            accuracy += 1
    
    return round(100 * (accuracy / len(lemmas_list)),2)



def testing_WSD_on_semcor(sentences_number, max_number_lemmas, iterations):
    """
        Esegue un numero di test di disambiguazione di sostantivi estratti 
        dal corpus semcor pari a "iteration". 
        Il parametro "sentence_number" indica il numero di frasi da estrarre 
        in modo casuale dal corpus semcor ad ogni test.
        Il parametro "max_number_lemmas" indica il massimo numero di lemmi da 
        disambiguare presi in modo casuale dalla lista di lemmi. 
        Se questo valore eccede il numero di lemmi in una lista, 
        viene selezionato un sottoinsieme casuale su tutta la lista.

    """
    
    total_accuracy = 0
    for i in range(iterations):
        print("iterazione numero " + str(i+1) + ":")
        tagged_sentences, sentence_list = open_random_semcor(sentences_number)
        lemmas_list = read_nouns_from_semcor(tagged_sentences)
        random_lemmas = lemmas_list
        max_elements = len(random_lemmas) if max_number_lemmas > len(random_lemmas) else max_number_lemmas
        
        # scelgo in modo casuale un insieme di lemmi da disambiguare
        if max_elements >= 2: 
            random_elements = random.randint(2,max_elements)
            print("numero di lemmi random: ", random_elements)
            random_lemmas = random.sample(random_lemmas, k=random_elements)
            accuracy = word_sense_disambiguation(random_lemmas, sentence_list)
            total_accuracy += accuracy
            print("accuratezza test " + str(accuracy) + "%\n")
        else:
            print("Nessun lemma NN etichettato con l'inicazione del wordnetsynset è stato trovato nel sottoinsieme di frasi del corpus semcor. Nessuna parola da disambiguare!\n")

    return round((total_accuracy / iterations),2)

