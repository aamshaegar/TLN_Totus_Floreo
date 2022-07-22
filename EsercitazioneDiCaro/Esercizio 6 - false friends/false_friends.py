import re
import nltk
import random
import numpy as np
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn

MAX_SENTENCES = 10000


def open_random_semcor(sentences_number):
    """
        Restituisce le prime "sentences_number" frasi scelte in modo casuale 
        da un insieme di corpus semcor annotati.
        La scelta avviene in un insieme di 37176 frasi, presenti nel corpus semcor
    """

    first_index = 0
    last_index = (MAX_SENTENCES - sentences_number)

    # indici di partenza e di fine del sottoinsieme di frasi da recuperare
    random_index = random.randint(first_index, last_index)
    last_element = random_index + sentences_number

    # recupero dal corpus Semcor un sottoinsieme di frasi le cui parole sono annotate con:
    #   - Lemma
    #   - POS
    #   - synset di wordnet
    
    sentences = semcor._items(None, "token", True, True, True)[random_index:last_element]
    return sentences



def read_content_word_from_semcor(tagged_sentences):
    """
        Legge da una lista di frasi tratte da un corpus semcor
        i lemmi associati alle parole di contenuto. Tali informazioni 
        verranno usate ai fini dell'individuazione dei false friends.
    """

    lemmas_list = []
    for i in range(len(tagged_sentences)):
        for couple in tagged_sentences[i]:

            # seleziono solo le parole di contenuto. Escludo parole non rilevanti
            _, _, lemma, pos, _, _ = couple
            if (lemma is not None) and (pos is not None):
                if (pos in ["n", "v", "s", "r"] and len(lemma) > 3):
                    lemmas_list.append(lemma)

    return lemmas_list




def random_couples(number_sentence, edit_threshold, similarity_threshold):
    """
        Questa funzione a partire da una lista di number_sentence frasi prese 
        dal corpus Semcor, recupera una lista di lemmi di termini presenti nelle
        frasi. In seguito calcola tutti gli accoppiamenti per ogni lemma. Rimuove 
        le coppie dupplicate e filtra per le coppie con una distanza di edit < di 
        edit_threshold.
        Infine calcola la similarità di wu and palmer tra i termini delle coppie 
        rimaste e restituisce quelle con una similarità minore di similarity_thrshold
    """

    # recupero la lista di parole dal corpus Semcor
    tagged_sentences = open_random_semcor(number_sentence)
    random_lemmas = read_content_word_from_semcor(tagged_sentences)

    # calcolo tutti i possibili accoppiamenti degli elementi
    all_couples = [(el1,el2) for el1 in random_lemmas for el2 in random_lemmas]

    # filtro le coppie per edit distance < di edit_threshold
    all_duplicates = [sorted(couple) for couple in all_couples if nltk.edit_distance(couple[0],couple[1]) < edit_threshold]
    
    # filtro gli accoppiamenti uguali
    not_equals = [couple for couple in all_duplicates if couple[0] != couple[1]]

    all_filtered = []
    all_similarity = []

    # rimuovo le coppie duplicate
    for couple in not_equals:
        if not couple in all_filtered:
            all_filtered.append(couple)

    # calcolo la similarità di wu and palmer sui termini nelle coppie
    for couple in all_filtered:
        sin1 = wn.synsets(couple[0])[0]
        sin2 = wn.synsets(couple[1])[0]
        similarity = wn.wup_similarity(sin1,sin2)

        # filtro per similarity_threshold
        if similarity < similarity_threshold:
            all_similarity.append((couple[0],couple[1],similarity))

    return all_similarity