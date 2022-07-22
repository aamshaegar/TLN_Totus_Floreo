from hanks import *
from collections import Counter
import sys

__name__ = "__main__"


def main(): 

    if len(sys.argv) < 2:
        print("\n***************************************")
        print("Error: Bad parameter!\nUSAGE: python main.py CORPUS")
        print("where, CORPUS is the path of the corpus which contains some sentences")
        print("***************************************\n")
        exit()
    
    sentences = read_file(sys.argv[1])
    semantic_couples = []


    for sent in sentences:

        # select fillers from parsing
        filler1, filler2 = dependency_parsing(sent)
        
        # disambiguate the term to extract its synset
        sub_synset, obj_synset = disambiguate_sentence(sent, filler1, filler2)
        
        # collect supersenses by the disambiguated terms
        sem1, sem2 = super_sense(sub_synset, obj_synset)
        
        # we define a semantic_types couples only if the supersenses are not None
        if sem1 is not None and sem2 is not None:
            semantic_couples.append((sem1,sem2))

        
        
    total_semantic_couples = len(semantic_couples)
    print("\ntotal number of sentences: ", len(sentences))
    print("total semantic types couples: ", total_semantic_couples)
    print()

    # count all occurrences of a single semantic couple
    # sorts all couples in descend order of frequency
    sc_count = Counter(semantic_couples)
    sorted_sc_count = sorted(sc_count.items(), key= lambda x:x[1], reverse = True)


    # return first 10 semantic couples with their frequency
    first_10 = [el for el in sorted_sc_count][:10]
    print("First 10 semantic types per frequency:")
    for sem_types in first_10:
        print("Semantic Types: ", sem_types)

    print()

    
    # return all the possible semantic types for the verb
    unique_semantic_couples = set(semantic_couples)
    print("Number of non duplicated semantic type couples found: ", len(unique_semantic_couples))
    print()
    print("All the semantic types found for the verb analized:")
    for sem_types in unique_semantic_couples:
        print("Semantic Types: ", sem_types)


if __name__ == "__main__":
    main()