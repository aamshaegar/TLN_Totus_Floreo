import nltk
import math
import numpy as np
from nltk.corpus import wordnet as wn

# Calcola la profondità massima di WordNet: 20
#MAX_DEPTH = max(max(len(hyp_path) for hyp_path in hypernym_paths(ss)) for ss in wn.all_synsets())
MAX_DEPTH = 20


def hypernym_paths(sense):
    """
        Restituisce la lista di tutti gli iperonimi sel senso dato in input.
        La funzione ricorsivamente cicla sugli iperonimi degli antenati del senso dato in input.
        
    """
    paths = []
    hypernyms = sense.hypernyms()

    # caso base. Non ci sono iperonimi, siamo alla radice dell'albero di 
    # appartenenza del senso dato in input. Restituiamo il senso.
    # La lista viene interpretata alla rovescia.
    if not hypernyms:
        return [[sense]]

    for hypernym in hypernyms:
        for ancestor_list in hypernym_paths(hypernym):
            ancestor_list.append(sense)
            
            # ricordiamo il giusto ordine della lista degli iperonimi dal senso ad entity.n.01
            ancestor_list.reverse()
            paths.append(ancestor_list)

    return paths



def depth(sense):
    """ Profondità massima di un senso """
    if not sense:
        return 0

    return max([len(path) for path in hypernym_paths(sense)])



def lowest_common_subsumer(syn1, syn2):
    """
        Calcola il primo antenato comune tra i sensi in input, se esiste, 
        altrimenti restituisce None
        L'antenato comune è calcolato considerando la lista di iperonimi dei sensi
        in input. Si prende la lista più lunga fra le possibili liste 
        di iperonimi dal senso alla radice del suo albero di wordnet.
        In seguito si scorrono le liste degli iperonimi dei due sensi e si individua, 
        se esiste il primo senso in comune, quindi lo si restituisce, altrimenti None.
        
    """

    # Lista dei percorsi che vanno dal senso "entity" (root di wordnet) al senso syn1 e syn2
    syn1_path = hypernym_paths(syn1)
    syn2_path = hypernym_paths(syn2)

    # Calcolo del percorso più lungo dei 2 synset    
    max_len_syn1_path = max(len(hyp_path) for hyp_path in syn1_path)
    max_len_syn2_path = max(len(hyp_path) for hyp_path in syn2_path)
    max_synset1_path = []
    max_synset2_path = []

    # Per ogni lista di synset in syn1_path
    for item in syn1_path:
        if len(item) == max_len_syn1_path:
            max_synset1_path = item

    for item in syn2_path:
        if len(item) == max_len_syn2_path:
            max_synset2_path = item


    # Per ogni synset 'sense' in max_synset1_path (partendo da entity) guardo se è presente in max_synset2_path

    lcs = None
    for sense in max_synset1_path:
        if sense in max_synset2_path:
            lcs = sense
    
    return lcs, max_synset1_path, max_synset2_path



def synsets_len(syn1, syn2):
    """
        Calcola la lunghezza del percorso, se esiste, tra i due sensi.
        Nello specifico, conta la distanza tra i nodi associati ai 2 
        sinset dati in input. La distanza è calcolata come somma del numero di nodi
        da attraversare dal syn1 fino al LCS tra syn1 e syn2, (se esiste) + 
        il numero dei nodi da attraversare dal syn2 al LCS tra syn1 e syn2.
    """
    
    if syn1 == syn2:
        return 0

    lcs, max_synset1_path, max_synset2_path = lowest_common_subsumer(syn1, syn2)
    if lcs is None:
        return None
    
    distance = 0
    for i in range(len(max_synset1_path)-1, -1, -1):
        if max_synset1_path[i] != lcs:
            distance += 1
        else:
            break
    
    for i in range(len(max_synset2_path)-1, -1, -1):
        if max_synset2_path[i] != lcs:
            distance += 1
        else:
            break

    return distance



def wu_and_palmer(syn1, syn2):
    """
        Metrica di similarità di wu_and_palmer.
        Il calcolo della similarità di wu_and_palmer vede l'individuazione 
        del sinset antentato comune (Lowest Common Subsumer) tra i sensi dati in input.
        Successivamente si definisce la profondità dai sensi in input alle radici dei loro 
        rispettivi alberi di wordnet.
        Il risultato del calcolo è il rapporto tra il doppio della lunghezza del percorso più 
        lungo dal LCS alla sua radice e la somma delle profondità dei sensi in input
    """

    lcs, max_synset1_path, max_synset2_path = lowest_common_subsumer(syn1, syn2)
    # Prendiamo tutti i percorsi che vanno dal senso "entity" (root di wordnet) al senso lcs se lcs esiste
    if lcs is None:
        return 0
    
    # Salviamo in max_len_lcs_path le lunghezza del percorso più lungo del synset lcs
    max_len_lcs_path = depth(lcs)


    # Formula di Wu & Palmer
    wupalmer = (2 * max_len_lcs_path) / (len(max_synset1_path) + len(max_synset2_path))
    return wupalmer

    

def shortest_path(syn1, syn2):
    """
        Metrica di similarità di shortest_path.
        Questa metrica si avvale della misura di distanza tra i synset dei 
        nodi che viene sotratta al doppio della profondità massima di wordnet. 
        La distanza è calcolata come somma del numero di nodi da attraversare dal 
        syn1 fino al LCS tra syn1 e syn2, (se esiste, altrimenti None) somato al
        numero dei nodi da attraversare dal syn2 al LCS tra syn1 e syn2.
        La profondità massima è calcolata come la lunghezza del percorso massimo fra 
        tutti i percorsi più lunghi di tutti i sensi in wordnet.
    """

    minP = synsets_len(syn1,syn2)

    # Formula di Shortest Path
    if minP is None:
        return 0
    else:
        return 2 * MAX_DEPTH - minP


def Leacock_Chodorow(syn1, syn2):
    """
        Metrica di similarità di Leacock e Chodorow.
        Anche questa metrica considera la distanza tra i synset dei nodi e la 
        profondità massima di wordnet.
    """

    minP = synsets_len(syn1,syn2)

    # Formula di Leacock & Chodorow
    if minP is None:
        return 0
    else:
        return -math.log((minP + 1) / (2 * MAX_DEPTH + 1))

