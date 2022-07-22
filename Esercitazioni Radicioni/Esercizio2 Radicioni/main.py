from FrameNetSynset import *
from os import path
    

__name__ = "__main__"

def main():

    prova = FrameNetDisambiguation("./data/annotazioni.json")
    total, exact, mean = prova.evaluation()
    print("Totale: ", total)
    print("Esatti: ", exact)
    print("Accuratezza: " + str(mean) + "%")

    students_frames = prova.get_students()
    for student in students_frames:
        
        name = student.get_student_name()
        if not path.exists("./output/" + name + ".txt"):
            file = open("./output/" + name + ".txt", "a", encoding="utf-8")
            
            frames = student.get_frame()
            fes = student.get_frame_element()
            lus = student.get_lexical_unit()

            file.write("***************** Disambiguate frames *****************\n")
            for frame in frames:
                file.write(str(frame))

            file.write("***************** Disambiguate frame elements *****************\n")
            for fe in fes:
                file.write(str(fe))

            file.write("***************** Disambiguate lexical units *****************\n")
            for lu in lus:
                file.write(str(lu))

            file.close()
            

if __name__ == "__main__":
    main()

