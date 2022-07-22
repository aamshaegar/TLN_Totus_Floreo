import nltk
from nltk.tokenize import TextTilingTokenizer
from nltk.corpus import brown
from matplotlib import pylab

with open('segmentation_dataset.txt') as f:
    dataset = f.readlines()
string_data = ''.join(dataset)
#print (string_data)

def demo(text=None):

    tt = TextTilingTokenizer(demo_mode=False)
    if text is None:
        text = brown.raw()[:1000]
        print(text)
    s, ss, d, b = tt.tokenize(text)
    pylab.xlabel("Sentence Gap index")
    pylab.ylabel("Gap Scores")
    pylab.plot(range(len(s)), s, label="Gap Scores")
    pylab.plot(range(len(ss)), ss, label="Smoothed Gap scores")
    pylab.plot(range(len(d)), d, label="Depth scores")
    pylab.stem(range(len(b)), b)
    pylab.legend()
    pylab.show()

print(demo(text=string_data))
