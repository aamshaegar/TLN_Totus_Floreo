from Similarity import *

__name__ = "__main__"

def main():

    term_sentences = read_csv_file("./data/definizioni.csv")
    preprocessing(term_sentences)

    similarities = dict()
    similarity_C_S = []     # Brick
    similarity_C_G = []     # Person
    similarity_A_S = []     # Revenge
    similarita_A_G = []     # Emotion

    total = 0
    for term in term_sentences:
        if not term in similarities:
            similarities[term] = []

        sentences = list(term_sentences[term])
        for i in range(0,len(sentences)):
            for j in range(i+1, len(sentences)):
                total += 1
                sim = bag_of_words_similarity(sentences[i], sentences[j])
                similarities[term].append(sim)

    
    similarity_threshold = 0.5

    # A similarity threshold is used to select only the couples of definition with a 
    # lexical overlap similarity greater than 0.5. We choice this value as an evidence  
    # of a similarity betweeen a couple of vector. 
    
    similarity_C_S = [sim for sim in similarities["Brick"] if sim >= similarity_threshold]
    similarity_C_G = [sim for sim in similarities["Person"] if sim >= similarity_threshold]
    similarity_A_S = [sim for sim in similarities["Revenge"] if sim >= similarity_threshold]
    similarita_A_G = [sim for sim in similarities["Emotion"] if sim >= similarity_threshold]

    # aggregate the similarities on the 4 dimension of 
    #   - concrete specific
    #   - concrete generic
    #   - abstract specific
    #   - abstract generic

    aggregate_sim_C_S = (100 * round(len(similarity_C_S) / total,2))
    aggregate_sim_C_G = (100 * round(len(similarity_C_G) / total,2))
    aggregate_sim_A_S = (100 * round(len(similarity_A_S) / total,2))
    aggregate_sim_A_G = (100 * round(len(similarita_A_G) / total,2))

    print("\nNumber of total couples: ", total)
    print("Percentage of the most similar couple for Brick (concrete/specific):", aggregate_sim_C_S , "%")
    print("Percentage of the most similar couple for Person (concrete/generic):", aggregate_sim_C_G, "%")
    print("Percentage of the most similar couple for Revenge (abstract/specific):",aggregate_sim_A_S, "%")
    print("Percentage of the most similar couple for Emotion (abstract/generic):", aggregate_sim_A_G, "%")
    print()



if __name__ == "__main__":
    main()