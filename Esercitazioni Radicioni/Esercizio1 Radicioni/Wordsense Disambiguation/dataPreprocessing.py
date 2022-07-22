import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Lista delle stopwords più comuni nella lingua inglese
STOP_WORDS = stopwords.words("english")
STOP_WORDS.append(',')
STOP_WORDS.append('.')
STOP_WORDS.append(';')
STOP_WORDS.append(':')
STOP_WORDS.append('etc')
STOP_WORDS.append("'s")
STOP_WORDS.append("Lt.")


lemmatizer = WordNetLemmatizer()

def bag_of_words_mapping(sentence):
    """
        procedura di pre-processing che costruisce a partire da una frase un insieme
        di termini rilevanti rimuovendo tutti i simboli di punteggiatura e le stopword
    """
    to_lower = lowerize(sentence)
    punctuation = remove_punctuation(to_lower)
    tokens = lemmatization(punctuation)
    stopwords_pure = remove_stopwords(tokens)
    return set(stopwords_pure)


def remove_stopwords(sentence_splitted):
    """ Rimuove le stopword da una lista di parole """

    return [value for value in sentence_splitted if value not in STOP_WORDS]



def lemmatization(sentence):
    
    """ Questa funzione esegue una lemmatizzazione delle parole date in input
        in una frase. L'operazione è condotta considerando il POS di ogni parola,
        della frase. Di questi, solo i sostantivi vengono considerati. 
        Successivamente, invocando la funzione lemmatize() della libreria 
        WordNetLemmatizer si ottiene il corretto lemma.       
    """

    sentence_splitted = []
    token = word_tokenize(sentence)
    for word, wtag in nltk.pos_tag(token):
        if pos_tag(wtag) == "n":
            sentence_splitted.append(lemmatizer.lemmatize(word))
    
    return sentence_splitted




def remove_punctuation(sentence):
    """ Rimuove tutti i simboli di punteggiatura e gli spazi multipli """   
    return re.sub('\s\s+', ' ', re.sub(r'[^\w\s]', '', sentence))



def lowerize(sentences):
    """ riduce in minuscolo tutte le maiuscole """
    return sentences.lower()


def pos_tag(tag):
    return tag[0].lower()