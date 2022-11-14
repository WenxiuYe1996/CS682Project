import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import PyPDF2
from nltk.stem import WordNetLemmatizer 

def get_doc_list(folder_name):
    doc_list = []
    doc_name = []
    file_list = [folder_name+'/'+name for name in os.listdir(folder_name) if name.endswith('txt')]
    for file in file_list:
        doc_name.append(file)
        st = open(file,'r').read()
        doc_list.append(st)
    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return doc_name, doc_list
 
def get_doc(folder_name):
    
    #read all the documents under given folder name
    doc_name, doc_list = get_doc_list(folder_name)

    print(doc_name)
    print(doc_list)
    
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
 
    taggeddoc = []
 
    texts = []
    for index,i in enumerate(doc_list):
        
        # for tagged doc
        wordslist = []
        tagslist = []
 
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        print(tokens)
 
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        print(stopped_tokens)
        print("--------------------------------------------------------------")
        # remove numbers
        number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        number_tokens = ' '.join(number_tokens).split()

        print(number_tokens)
        print("--------------------------------------------------------------")
        # stem tokens
        lemmatized_tokens = [lemmatizer.lemmatize(i) for i in number_tokens]
        print(lemmatized_tokens)
        print("--------------------------------------------------------------")
        
        # remove empty
        length_tokens = [i for i in lemmatized_tokens if len(i) > 1]
        # add tokens to list
        texts.append(lemmatized_tokens)
        
 
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(lemmatized_tokens))).split(),str(index))
        taggeddoc.append(td)
        print(taggeddoc)
 
    return doc_name, taggeddoc


 
doc_name, taggeddoc = get_doc('txt')
print ('Data Loading finished')
 
# build the model
model =  gensim.models.doc2vec.Doc2Vec(taggeddoc, vector_size=30, min_count=1, epochs=80)


print(len(taggeddoc))

for i in range(0, 2):
    for j in range(0, 2):
        print(f"i:  {i}, j : {j}")
        if (i!=10 and i != 9):
            print(model.dv.similarity(i, j))
            print("------------------------------")