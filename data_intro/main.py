from nltk.corpus import universal_treebanks

from preprocessing import *

if __name__ == '__main__':
    result = processData('dataset/Train.csv')
    print(result)
    #[table, chair, computer]
    #[0, 2, 1]
    #i sat on my chair and turned on my computer

