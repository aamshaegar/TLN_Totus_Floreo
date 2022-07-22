from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from dataPreprocessing import *
import nltk.wsd
import re
import nltk



def create_context(sense):
    """
        Costruisce il contensto per i singoli sensi esaminando le definizioni e le 
        glosse del synset e quelle dei suoi iponimi e iperonimi diretti.
    """

    signature = []
    signature += sense.examples()
    signature += [sense.definition()]
    
    # considero tutte le definizioni e gli esempi degli iponimi e gli iperonimi di sense
    for hyp in sense.hypernyms():
        signature += hyp.examples()
        signature += [hyp.definition()] 

    for hyp in sense.hyponyms():
        signature += hyp.examples()
        signature += [hyp.definition()]

    return bag_of_words_mapping(" ".join(signature))

    
    
def lesk(word, sentence):
    """
        Algoritmo di Lesk. Le features estratte, utili ai fini del processo 
        di disambiguazione, considerano l'intersezione trai lemmi del contesto 
        e i lemmi delle definizioni e degli esempi dei sinset e dei loro iponimi e 
        iperonimi della parola in input.

        Args:
            word(str): la parola da disambiguare
            sentence(str) una frase da contesto per dismbiguare
        
        Return:
            best_sense: (synset) il migliore senso per la parola da disambiguare nel contesto
    """
    
    synsets = wn.synsets(word)
    best_sense = []
    if synsets:
        
        best_sense = synsets[0]
        max_overlap = 0
        context = bag_of_words_mapping(sentence)

        for sense in synsets:          
            signature = create_context(sense)
            overlap = len(context.intersection(signature))

            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = sense
        
    return best_sense
