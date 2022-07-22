import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from dataPreprocessing import *
from pprint import pprint
import pyLDAvis
import sys
import pyLDAvis.gensim_models


__name__ = "__main__"

def main():


    if len(sys.argv) < 2:
            print("\n***************************************")
            print("Error: Bad parameter!\nUSAGE: python main.py DATASET")
            print("where, DATASET is the path of the file which contains some sentences")
            print("***************************************\n")
            exit()

    # read the document
    dataset_path = sys.argv[1]
    document = open_and_check(dataset_path)

    # lemmatize and tokenize all sentences
    tokenized_sentence = []
    for sent in document:
        tokens = list(bag_of_words_mapping(sent))
        tokenized_sentence.append(tokens)



    # Dictionary: Association between token and unique id
    id2word = corpora.Dictionary(tokenized_sentence)

    # Corpus: Create association betweeen term and its paragrafp location in the document
    corpus = [id2word.doc2bow(text) for text in tokenized_sentence]


    # Human readable format of corpus (term-frequency)
    corpus_readable = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
    print("\n corpus: ")
    pprint(corpus_readable)
    print()

    # build graph
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word,num_topics=4)
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, './output/LDA_Visualization.html')


if __name__ == "__main__":
    main()