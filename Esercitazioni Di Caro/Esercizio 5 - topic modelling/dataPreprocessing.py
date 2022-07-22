import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


STOP_WORDS = stopwords.words("english")
STOP_WORDS.append(',')
STOP_WORDS.append('.')
STOP_WORDS.append(';')
STOP_WORDS.append(':')
STOP_WORDS.append('etc')
STOP_WORDS.append("'s")
STOP_WORDS.append("Lt.")
STOP_WORDS.append('\n')
STOP_WORDS.append('\t')
STOP_WORDS.extend(['from', 'subject', 're', 'edu', 'use'])



def bag_of_words_mapping(sentence):
    """
        from a sentence, (string) remove all punctuation symbols, tokenize the sentence 
        into a list of words. Finally remove all stopwords from the returned list.
    """
    punctuation = remove_punctuation(sentence)
    tokens = lemmatization(punctuation)
    stopwords_pure = remove_stopwords(tokens)
    return set(stopwords_pure)



def remove_stopwords(sentence_splitted):
    """
        from a list of well noted stopwords, remove all the stopwords
        from a list of words passed as parameter
    """
    return [value for value in sentence_splitted if value not in STOP_WORDS]



def lemmatization(sentence):
    """ Lemmatize a sentence passed as parameter """

    sentence_splitted = []
    lemmatizer = WordNetLemmatizer()
    for tag in nltk.pos_tag(word_tokenize(sentence)):
        sentence_splitted.append(lemmatizer.lemmatize(tag[0]).lower())
    return sentence_splitted



def remove_punctuation(sentence):
    """ Remove punctuation and multiple spaces from a string """

    return re.sub('\s\s+', ' ', re.sub(r'[^\w\s]', '', sentence))



def tokenize_sentence(sentence):
    """ tokenize without doing bag of words """

    return word_tokenize(sentence)



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