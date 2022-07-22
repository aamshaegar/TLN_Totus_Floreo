from DataPreprocessing import *
from collections import Counter
from nltk.corpus import wordnet as wn


def open_and_check(dataset_path):
    """
        Open a file and read all lines.
        - Throw FileNotFoundError if file not exists
        - Throw ValueError if file is void
        - param "dataset_path": path of a file
    """

    try:
        file = open(dataset_path, "r", encoding='utf8')
    except FileNotFoundError:
        print("Error! File not found...")
        exit()

    lines = file.readlines()
    if lines == []:
        raise ValueError("Error, void file in input!")

    file.close()
    return lines



def read_csv_file(csv_path):
    """
        read from a comma separated file all the data used for calculate similarities
        param "csv_path": path of a csv file
    """

    lines = open_and_check(csv_path)
    terms = dict()

    for line in lines:
        sentence_splitted = line.split(",")
        if not sentence_splitted[0] in terms:
            terms[sentence_splitted[0]] = sentence_splitted[1:]
                
    return terms



def preprocess_definition(terms):
    """
        For all the definition of a single term, 
        this function creates a bag of words replacing
        the list of term by a set of relevant words.
        - param "terms": dictionary of associations term - definitions
    """
    for term in terms:
        for i in range(0,len(terms[term])):
            sentence = bag_of_words_mapping(terms[term][i])
            terms[term][i] = sentence



def most_frequent(context):
    """
        This function retrieve the most frequent term in all the denifitions of a concept.
        - param "context": all the preprocessed definitions of a concept
    """
    
    context_count = Counter(context)
    sorted_context = sorted(context_count.items(), key= lambda x:x[1], reverse = True)
    most_frequent = sorted_context[0][0]
    
    return most_frequent




def context_for_synset(synset):
    """
        Return the context of a WordNet Synset using all the definitions
        and the examples of the synset and its hyponyms
        - param "synset": a wordnet synset
    """
    
    context = set()
    context.update(bag_of_words_mapping(synset.definition()))
    for example in synset.examples():
        context.update(bag_of_words_mapping(example))


    for hyponym in synset.hyponyms():
        context.update(bag_of_words_mapping(hyponym.definition()))
        for example in hyponym.examples():
            context.update(bag_of_words_mapping(example))

    return context



def best_score(context, genus):
    """
        Compute the best synset using the lexical overlap between the context 
        from all the definitions and the context from the main genus synset 
        and its direct hiponyms.
        Score is computed using bag of words's approach
        - param "context": the context retrieved by the bag of words of all the definitions
        - param "genus": the most frequent term from the definitions. 
    """

    
    # lif of all the hyponyms of the synsets of the genus
    list_synsets = []
    synsets = wn.synsets(genus)

    for syn in synsets:
        list_synsets += syn.hyponyms()
    
    # a list of all the hyponyms of the synsets of the genus with its score
    best_synsets = []

    for synset in list_synsets:
        synset_context = context_for_synset(synset)
        score = len(context & synset_context) + 1 
        best_synsets.append((synset, score))
    
    return best_synsets
