import json
import re
import nltk
STOP_WORDS = stopwords.words("english")


def bag_of_words(sentence):
    """
        from a sentence, (string) remove all punctuation symbols, tokenize the sentence 
    """
    
    stopwords_pure = remove_stopwords(tokens)
    return set(stopwords_pure)


def remove_stop(sentence_splitted):

    return [value for value in sentence_splitted if value not in STOP_WORDS]




def clean_lu_name(lexical_unit):
    """
        Preprocessing of a lexical unit name. 
        Lexical unit names are in the form <lu>.PoS. This function eliminate the .POS 
        substring from a lexical unit name.

    """
    return lexical_unit.split('.')[0]


def clean_lu_definition(lu_definition):
    """
        Preprocessing of a lexical unit definition.
        Lexical unit definitions are in the form <type>: definition. This function eliminate
        the <type>: substring from a lexical unit definition
    """
    return lu_definition.split(':')[1]
