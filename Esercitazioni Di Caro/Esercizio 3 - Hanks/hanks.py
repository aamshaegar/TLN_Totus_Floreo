from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from data_preprocessing import *
import spacy


# The default model for the English language is en_core_web_sm. 
nlp = spacy.load('en_core_web_sm')

# Syntactic dependency relation for each subject and object
POSSIBLE_SUBJ = ['subj', 'nsubjpass', 'nsubj']
POSSIBLE_OBJ = ['pobj', 'dobj', 'obj', 'iobj']

# subj = subject
# nsbjpass = passive subject indirect
# nsubj = indirect subject
#
# obj = object
# dobj = direct object
# pobj = predicative object
# iobj = indirect object

# in una frase passiva con verbo valenza 2 c'è un nsubjpass e un pobj
# un oggetto predicativo è un oggetto che non è legato alla ROOT ma ad una preposizione


# resolve all pronouns: example
#ALL_PERSON = ["i", 'you', 'he', "she", "it", "we", "they", "who"]
NOUNS = ["-PRON-", "PROPN", "PRON"]
SYNSET_PERSON = wn.synsets("Person")[0]


# mapping pos tag to lesk pos tag
def pos_mapping(pos):
    return pos[0].lower()


def super_sense(synset1, synset2):  
    """ return the supersense of the synset in input """
      
    syn1 = synset1
    syn2 = synset2
    
    if synset1 is not None:
        syn1 = syn1.lexname().split(".")[1]
    
    if synset2 is not None:
        syn2 = syn2.lexname().split(".")[1]

    return syn1, syn2



def dependency_parsing(sentence):
    """
        Perform a dependency parsing of the sentences readed from the corpus.
        term is the verb on which it selects all the slot for subject and object
    """

    parsing_sentence = nlp(sentence)
    subj = None   # couple with name, pos 
    obj = None

    for token in parsing_sentence:

        if token.head.lemma_ == "love":
            if token.dep_ in POSSIBLE_SUBJ and subj is None:
                subj = (token.lemma_, token.pos_)

            if token.dep_ in POSSIBLE_OBJ and obj is None:
                obj = (token.lemma_, token.pos_)

    return subj, obj    



def disambiguate_sentence(context, subject, object_):
    """
        Execute a disambiguation on the subject and the object in input 
        using the lesk algorithm. The context is the sentence which 
        subjs and objs occurs
    """
    synset_subj = None
    synset_obj = None

    if subject is not None:
        if subject[0] in NOUNS or subject[1] in NOUNS: synset_subj = SYNSET_PERSON
        else: synset_subj = lesk(context, subject[0], pos_mapping(subject[1]))

    if object_ is not None:
        if object_[0] in NOUNS or object_[1] in NOUNS: synset_obj = SYNSET_PERSON
        else: synset_obj = lesk(context, object_[0], pos_mapping(object_[1]))

    """
    if synset_obj is None and synset_subj is None:
        print(context, subject, object_, synset_subj, synset_obj)
    """

    return synset_subj, synset_obj
