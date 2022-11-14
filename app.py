from bs4 import BeautifulSoup
import requests
import re
import pandas

from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import numpy as np
import PyPDF2
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
# NLTK Stop words
from nltk.corpus import stopwords
import string
import warnings
warnings.simplefilter('ignore')
from nltk.stem.wordnet import WordNetLemmatizer
from pprint import pprint
import gensim
import gensim.corpora as corpora


def extractTextFromScannedPDFs(fname):

    #code to extract text from scanned pdfs
    pages = convert_from_path(fname)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)

    new_text = text.strip("\n")
    return new_text


def extractTextFromPDFs(fname):
    #code to extract text from pdfs
    pdfFileObj = open(fname, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    text = ""

    for i in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(i)
        text += pageObj.extractText()
    pdfFileObj.close()

    return text

def getTextFromNasaWeb():

    input_topic = input("What topic do you want to search for? ")
    input_topic_split = input_topic.split()
    key_words=""
    for word in input_topic_split:
        key_words+=word
        key_words+="%20"

    #remove last three charater
    key_words = key_words[:-3]
    print(key_words)

    #url for nasa website
    url = f"https://ntrs.nasa.gov/search?q={key_words}"
    #numb of pange 
    url += "&page=%7B%22size%22:10000,%22from%22:0%7D"
    print(url)

    #use BeautifulSoup to get the html information
    page = requests.get(url)
    doc = BeautifulSoup(page.text, "html.parser")


    c = doc.find('script', id="serverApp-state").text
    #print(c)
    result_list = re.findall('&q;/(.+?)&q;', c)
    res = []
    #[res.append(x) for x in result_list if x not in res and 'txt' not in x and 'xlsx' not in x and 'pptx' not in x and 'docx' not in x]
    [res.append(x) for x in result_list if x not in res and 'txt' not in x and 'xlsx' not in x and 'pptx' not in x and 'docx' not in x and 'mp4' not in x]
  
    pdf_link_list = []
    link_list = []
    for x in res:
        pdf_link_list.append(f"https://ntrs.nasa.gov/{x}")
        link = x.split("/", 1)[1]
        link = link.split("/downloads")[0]
        link_list.append(f"https://ntrs.nasa.gov/{link}")

    print(f"length pdf link:{len(pdf_link_list)}")
    for v in pdf_link_list:
        print(v)
    
    text_list = []
    pdf_path = []
    #download the pdfs
    print(f"length link :{len(pdf_link_list)}")
    i = 0
    for p in pdf_link_list:
        print(f"i:{i}")
        response = requests.get(p)
        loc = res[i].split("downloads/")[1]
        print(loc)
        path = f"pdf/{loc}"
        open(path, "wb").write(response.content)
        print(path)
        text = getText(path)
        text_list.append(text)
        pdf_path.append(path)

        new_text = text.strip("\n")
        #print(new_text)


        with open(f'txt/{key_words}{i}.txt', 'w') as f:
            f.write(new_text)
            f.close() 
        i+=1
        # creating df object with columns specified    
        

    print(pdf_path)
    #return pdf_path

def getText(path):

    text = extractTextFromPDFs(path)
    print("text length:" )
    print(len(text))
    if (len(text) < 5):
        print("<5")
        text = extractTextFromScannedPDFs(path)
    return text

def cleanText():
    df = pd.DataFrame(text_list, columns =['text']) 
    print(df)

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(text):
        stop_free = ' '.join([word for word in text.lower().split() if word not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = ' '.join([lemma.lemmatize(word) for word in punc_free.split()])
        return normalized.split()

    new_list = []
    new_list = df['text'].apply(clean)

    df['text_clean'] = df['text'].apply(clean)

    #print(df)

    #df.to_csv(f"/Users/wenxiuye/Desktop/clean_code/csv/data.csv", header=True)




getTextFromNasaWeb()


#df = pd.DataFrame(pdf_path, columns =['path']) 
#df.to_csv(f"/Users/wenxiuye/Desktop/topic_modeling/csv/pdf_path.csv", header=True) 
#df = pd.read_table(f"/Users/wenxiuye/Desktop/topic_modeling/csv/pdf_path.txt", delimiter=" ")
  
# display DataFrame
#print(df.path)