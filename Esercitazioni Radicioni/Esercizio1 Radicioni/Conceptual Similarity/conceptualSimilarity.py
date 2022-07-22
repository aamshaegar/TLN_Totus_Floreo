import pandas as pd
import math
from similarity import *
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn


class conceptualSimilarity:

    def __init__(self, cvs_path = None):
        self.__similarity_result = []
        self.__similarity = []
        self.__number_row = 0
        self.__word1 = []
        self.__word2 = []
        self.__float = []

        if not cvs_path is None:
            self.__read_csv(cvs_path)



    def __read_csv(self, cvs_path):
        """
            Questa funzione legge dal file wordSin 353. Il file in input 
            deve essere formato da 3 colonne:
                - word 1: un termine
                - word 2: un termine
                - Human (mean): la similarità tra word1 e word2
        """
        
        pd.options.display.max_rows = 1000
        couple = pd.read_csv(cvs_path)
        self.__word1 = couple['Word 1']
        self.__word2 = couple['Word 2']
        self.__float = couple['Human (mean)']
        self.__number_row = len(self.__word1.iloc[:])




    def calculate_similarity(self, similarity_metric):
        """
            Questa funzione, data in input una metrica di similarità, scelta tra:
                - wupalmer: wu and palmer
                - shortest: shortest_path
                - leacock: leacock e Chodorow
            restituisce la similarità per ogni coppia di termini Word1 Word2 letti dal
            file .csv, come massimo della similarità tra ogni coppia di sensi dei 2 termini.
        """
        
        for i in range(self.__number_row):
            term1 = self.__word1.iloc[i]
            term2 = self.__word2.iloc[i]
            wordSin = self.__float.iloc[i]
            similarity = self.max_similarity(term1,term2,similarity_metric)[0]
            self.__similarity_result.append(similarity)
            self.__similarity.append((term1, term2, similarity, wordSin))

        return self.__similarity_result




    def max_similarity(self, term1, term2, similarity_metric):
        """
            Funzione di utilità.
            Serve per calcolare il massimo della similarità tra ogni coppia di sensi dei 2 termini "term1 e term2"
        """

        if not similarity_metric in ["wupalmer", "shortest", "leacock"]:
            raise ValueError("Error, incorrect using of similarity_metrics parameter\nUSAGE: one of ['wupalmer','shortest','leacock']")
        
        syns1 = wn.synsets(term1)
        syns2 = wn.synsets(term2)
        max_similarity = 0
        max_synsets = ()
        similarity = 0
        
        for syn1 in syns1:
            for syn2 in syns2:
            
                # switch to similarity metrics
                if similarity_metric == "wupalmer":
                    similarity = wu_and_palmer(syn1,syn2)
                elif similarity_metric == "shortest":
                    similarity = shortest_path(syn1, syn2)
                elif similarity_metric == "leacock":
                    similarity = Leacock_Chodorow(syn1,syn2)
            
                if(max_similarity < similarity):
                    max_similarity = similarity
                    max_synsets = (syn1,syn2)
        
        return max_similarity, max_synsets




    def similarity_correlation(self, correlation):
        """
            Calcola la correlazione (pearson o spearman) tra le similarità presenti nel file.csv e quelle 
            restituite dal calcolo della similarità con le metriche di wordnet.
            correlation prende parametri in: ["pearson", "spearman"].
        """

        if not correlation in ["pearson", "spearman"]:
            raise ValueError("Error, incorrect using of similarity_metrics parameter\nUSAGE: one of ['pearson','spearman']")
        
        if self.__similarity_result == []:
            raise Exception("Error, calculate similarity, first!")

        sim_cor = 0
        if correlation == "pearson":
            sim_cor = pearsonr(self.__float.iloc[:], self.__similarity_result[:])[0]
        elif correlation == "spearman":
            sim_cor = spearmanr(self.__float.iloc[:], self.__similarity_result[:])[0]

        return sim_cor
                

    def get_similarity(self):
        return self.__similarity

