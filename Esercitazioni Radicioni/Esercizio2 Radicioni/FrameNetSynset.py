import json
import nltk
from nltk.corpus import framenet as fn
from utility_class import *


class FrameNetDisambiguation:
    """
        Read the annotation of all the frame terms present in a Json file and 
        perform a wordsense disambiguation task.
    """

    def __init__(self, json_file_annotation):

        self.__annotation = self.__read_json_annotation(json_file_annotation)
        self.__students_frames = []
        self.__init_class_frame_synsets()


    def __read_json_annotation(self, json_file_annotation):        
        """
            Open a json file containing all frame annotations per student
            Throw FileNotFoundError if file not exists
        """

        try:
            file = open('data/annotazioni.json')
        except FileNotFoundError:
            print("Error! File not found...")
            exit()
        
        json_file = json.load(file)
        if json_file == []:
            raise ValueError("Error, void dataSet in input!")

        file.close()
        return json_file

    
    def __init_class_frame_synsets(self):
        """
            Read the json file and organize the annotation per student.
            Take from framenet all the definition of the terms defined in the file. 
            Execute a preprocessing on the contexts of the terms to perform, in 
            future, a wordsense disambiguation task. 
        """

        for student in self.__annotation:
            new_student = StudentFrames(student)

            # Encapsulation for single frame
            for frame in self.__annotation[student]:

                # check if exists multiword expressions
                name = frame["frame_name"][0]
                term_to_disambiguate = frame["frame_name"][1]
                synset = frame["wordnet_synset"]
                
                frame_from_framenet = fn.frame_by_name(name)
                definition = frame_from_framenet.definition
                bag_of_word_definition = bag_of_words_mapping(definition)

                term = DisambiguateTerm(term_to_disambiguate, synset, bag_of_word_definition, Term.frame)
                new_student.set_frame(term)

                
                # Encapsulation for every frame elements of a single frame
                for frame_element in frame["frame_elements"]:
            
                    term_to_disambiguate = frame_element
                    synset = frame["frame_elements"][frame_element]

                    if "_" in frame_element:
                        term_to_disambiguate = frame["frame_elements"][frame_element][0]
                        synset = frame["frame_elements"][frame_element][1]

                    definition = frame_from_framenet.FE[frame_element].definition
                    bag_of_word_definition = bag_of_words_mapping(definition)
                    
                    term = DisambiguateTerm(term_to_disambiguate, synset, bag_of_word_definition, Term.frame_element)
                    new_student.set_frame_element(term)


                # Encapsulation for every lexical units of a single frame
                for lexical_unit in frame["lexical_units"]:

                    term_to_disambiguate = clean_lu_name(lexical_unit)
                    synset = frame["lexical_units"][lexical_unit]

                    if " " in lexical_unit:
                        term_to_disambiguate = frame["lexical_units"][lexical_unit][0]
                        synset = frame["lexical_units"][lexical_unit][1]

                    definition = frame_from_framenet.lexUnit[lexical_unit].definition
                    bag_of_word_definition = bag_of_words_mapping(clean_lu_definition(definition))
                  
                    term = DisambiguateTerm(term_to_disambiguate, synset, bag_of_word_definition, Term.lexical_unit)
                    new_student.set_lexical_unit(term)


            self.__students_frames.append(new_student)



    def __is_student(self, student_name):
        """ check  if student_name is associated to some frame in the json annotation file """

        student_found = None
        for student in self.__students_frames:
            if student.get_student_name() == student_name:
                student_found = student
        
        if student_found is None:
            raise ValueError("Error, student '" + student_name + "' doesn't exist!")

        return student_found



    def get_students(self):
        """ Return all the student object in the json annotation file """
        return [el for el in self.__students_frames]



    def get_frame_from_student(self, student_name):
        """ Return, if exist, the student 'student_name' read in the json annotation file """
        try:
            student_frame = self.__is_student(student_name)
            return student_frame.get_frame()
        except ValueError as e:
            print(e)


    def get_frame_element_from_student(self, student_name):
        """ Return the list of the frame_element of all the frame associated 
            to the student student_name, if exist, read in the json annotation file 
        """
        try:
            student_frame = self.__is_student(student_name)
            return student_frame.get_frame_element()
        except ValueError as e:
            print(e)


    def get_lexical_unit_from_student(self, student_name):
        """ Return the list of the lexical unit of all the frame associated 
            to the student student_name, if exist, read in the json annotation file 
        """
        try:
            student_frame = self.__is_student(student_name)
            return student_frame.get_lexical_unit()
        except ValueError as e:
            print(e)


    def evaluation_per_student(self, student_name):
        """ 
            Check if student_name exists in the list of students, then return 
            the resulting match between the synsets annotated and the synsets 
            obtained from the disambiguation process
        """

        total = 0
        exact = 0
        student_frame = None
        try:
            student_frame = self.__is_student(student_name)
        except ValueError as e:
            print(e)

        frames = student_frame.get_frame()      # get all frames disambiguated
        fe = student_frame.get_frame_element()  # get all frame elements disambiguated
        lu = student_frame.get_lexical_unit()   # get all lexical units disambiguated

        # count all the terms for the student "student_name"
        total += len(frames)
        total += len(fe)
        total += len(lu)

        # check all the exact match between the annotation and the sysnsets obtained
        exact += len([el for el in frames if el.compare() == 1])
        exact += len([el for el in fe if el.compare() == 1])
        exact += len([el for el in lu if el.compare() == 1])

        return total, exact, 100 * round((exact/total), 2)


    def evaluation(self):
        """
            return all the resulting match between the synsets annotated and the synsets 
            obtained from the disambiguation process for all the students
        """

        total = 0
        exact = 0
        students = self.get_students()
        for s in students:
            tot, ex, _ = self.evaluation_per_student(s.get_student_name())
            total += tot
            exact += ex

        return total, exact, 100 * round((exact/ total), 2)



    def print_all_frame_information(self):
        """ print all the information from the StudentFrames objects """

        for student in self.__students_frames:
            print(student)

