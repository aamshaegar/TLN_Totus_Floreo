from onomasiologic_research import *


__name__ = "__main__"

def main():

    # open and replace with a bag of words of terms every list of definitions
    definitions = read_csv_file("./data/definizioni.csv")
    preprocess_definition(definitions)

    
    frequent_terms = dict()     # for each concept retrieve the most frequent term in the denifitions
    context_terms = dict()      # for each concept define a bag of words of the terms in all the definitions

    for concept in definitions:
        
        context = []
        if not concept in frequent_terms:
            frequent_terms[concept] = []

        if not concept in context_terms:
            context_terms[concept] = []

        for definition in definitions[concept]:
            context += list(definition)

        #retrieve the most frequent term in the denifitions
        terms = most_frequent(context)
        frequent_terms[concept] = terms
        context_terms[concept] = set(context)



    for concept in frequent_terms:

        genus = frequent_terms[concept]
        print("\nOnomasiologic research for the concept: ", concept)
        print("Genus found: ", genus)

        best_synsets = best_score(context_terms[concept], genus)
        sorted_by_score = sorted(best_synsets, key=lambda x:x[1], reverse=True)
        top_5_synsets = [el[0] for el in sorted_by_score][:5]
        
        print("\nSynsets found:")
        for syn in top_5_synsets:
            print("- " + str(syn))
        print("\n")

    

if __name__ == "__main__":
    main()