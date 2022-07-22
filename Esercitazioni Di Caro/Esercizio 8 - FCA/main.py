import os
import sys
import random
import concepts
import graphviz
from formal_concept_analisys import *
from concepts import Context
from pathlib import Path

#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

__name__ = "__main__"

def main():

    
    if len(sys.argv) < 3:
        print("\n***************************************")
        print("Error: Bad parameter!\nUSAGE: python main.py DATASET METHOD")
        print("where, DATASET is the path of the file which contains some sentences")
        print("where, METHOD is the method of the built of the FCA lattice. One of [single,all]")
        print("\t Use 'single' for built the lattice on a randomic sentence on the file.")
        print("\t Use 'all' for analize the whole file.")
        print("***************************************\n")
        exit()


    dataset_path = sys.argv[1]
    method = sys.argv[2]

    if method not in ["single", "all"]:
        print("\n***************************************")
        print("Error: Bad using of METHOD parameter!")
        print("Select one of [single,all]")
        print("***************************************\n")
        exit()

    document = open_and_check(dataset_path)
    tokenized_document = document_sentence_tokenize(document)
    
    if method == "single":
        tokenized_document = [random.choice(tokenized_document)]
        print("\nSentence analized: ", tokenized_document[0])
    
    # select all content words and their list of sintactic dependencies
    words_by_pos = sentence_parsing(tokenized_document)

    # build adiacent matrix
    file_name = Path(dataset_path).stem
    csv_file_path = "./csv/" + 'matrix_' + file_name + ".csv"
    lattice_file_path = "lattice_" + file_name
    build_adiacent_matrix(csv_file_path, words_by_pos)
    fca = concepts.load_csv(csv_file_path)  
    print("Adiacent matrix")
    print(fca)
    print()
    fca = fca.lattice.graphviz(directory="./output", view=True, filename= lattice_file_path)
    print("Formal context")
    print(fca)

if __name__ == "__main__":
    main()