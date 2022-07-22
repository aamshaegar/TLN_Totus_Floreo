import enum
from nltk.corpus import wordnet as wn 
from DataPreprocessing import *


class Term(enum.Enum):
    frame = 1
    frame_element = 2
    lexical_unit = 3


class StudentFrames:
    """
        StudentFrames is a class for recording all the frame, frame elements and 
        lexical units associated to a student name.
        The class is a sort of container for the information extracted during the
        reading of the file.
        Each of the list objects in the __init__ is a list of a generic Term
    """

    def __init__(self, student_name):

        self.__student_name = student_name
        self.__frame = []
        self.__frame_element = []
        self.__lexical_unit = []
        

    def get_student_name(self):
        return self.__student_name


    def get_frame(self):
        return self.__frame


    def get_frame_element(self):
        return self.__frame_element


    def get_lexical_unit(self):
        return self.__lexical_unit
        

    def set_frame(self,frame):
        self.__frame.append(frame)


    def set_frame_element(self,fe):
        self.__frame_element.append(fe)


    def set_lexical_unit(self,lu):
        self.__lexical_unit.append(lu)


    def __str__(self):
        ris = "Frames for student: " + self.__student_name + "\n"
        for frame in self.__frame:
            ris += str(frame) + "\n"

        ris += "Frame elements \n"
        for fe in self.__frame_element:
            ris += str(fe) + "\n"

        ris += "Lexical Units \n"
        for le in self.__lexical_unit:
            ris += str(le) + "\n"

        return ris


class DisambiguateTerm:
    """
        Utility class, used to perform the disambiguation of the frame, 
        frame elements and lexical unit. All of this elements are interpretaded as 
        generics "Term" to disambiguate. 
        The method 'best_score' calculates the best sense for every term.
    """

    def __init__(self, name, synset_annotation, definition, term_type):
        self.name = name
        self.synset_annotation = synset_annotation
        self.definition = definition
        self.best_synset = ""
        self.result = ""
        self.type = term_type
        self.__best_score()
  

    def __context_for_synset(self, synset):
        """
            Return the context of a WordNet Synset using definition, examples, hypernyms and hyponyms
        """
        
        context = set()

        context.update(bag_of_words_mapping(synset.definition()))
        for example in synset.examples():
            context.update(bag_of_words_mapping(example))
        
        for hypernym in synset.hypernyms(): 
            context.update(bag_of_words_mapping(hypernym.definition()))
            for example in hypernym.examples():
                context.update(bag_of_words_mapping(example))
                
        for hyponym in synset.hyponyms():
            context.update(bag_of_words_mapping(hyponym.definition()))
            for example in hyponym.examples():
                context.update(bag_of_words_mapping(example))

        return context


    def __best_score(self):
        """
            Compute best synset intersecting FrameNet context and WordNet context
            Score is computed using bag of words's approach
        """

        synsets = wn.synsets(self.name)
        if not synsets:
            return None

        best_synset = synsets[0]
        max_score = 0

        for synset in synsets:
            synset_context = self.__context_for_synset(synset)
            score = len(self.definition & synset_context) + 1 
            if score > max_score:
                max_score = score
                best_synset = synset
        
        self.best_synset = best_synset
        return best_synset


    def compare(self):
        synset = str(self.best_synset)
        syn_string = synset.split("'")[1]
        if self.synset_annotation == syn_string:
            self.result = "Correct!"
            return 1
        else:
            self.result = "Wrong!"
            return 0


    def __str__(self):

        ris = "" 
        if self.type == Term.frame: ris += "frame"
        elif self.type == Term.frame_element: ris += "frame element"
        else: ris += "lexical unit"

        ris += ": " + self.name + ":\n - "
        ris += "Synset annotation: " + self.synset_annotation + "\n - " 
        ris += "Best Synset: " + str(self.best_synset) + "\n - "
        ris +=  self.result + "\n"
        ris += "-" * 30
        ris += "\n"
        return ris
