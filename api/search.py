from bs4 import BeautifulSoup
import requests
import re
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import PyPDF2
import pandas as pd
import time
import shutil
from IPython.display import HTML
from requests.exceptions import ConnectTimeout
import urllib.request
import io
import pandas as pd
import csv
import os
import pytesseract
from pdf2image import convert_from_path
import shutil
import numpy as np
import nlp
# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.stem.porter import *
# spacy for lemmatization
import spacy
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import pprint
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import PyPDF2
from nltk.stem import WordNetLemmatizer 
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're','in','for', 'and', 'of','the', 'is', 'edu', 'to', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

# Web Scraper for NASA website
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
 #Create new directory for given topic
def makeNewDiretoryForGivenTopic(userInputTopic):
    
    dirName = userInputTopic.replace(" ", "")

    pdfDirName = f"pdf{dirName}"
    csvDirName = f"csv{dirName}"
    txtDirName = f"txt{dirName}"
    ldaDirName = f"lda{dirName}"

    parent_dir = ""

    if os.path.exists(pdfDirName):
        print("directory already exits")
        #shutil.rmtree(pdfDirName)
    else:
        path = os.path.join(parent_dir, pdfDirName)
        os.mkdir(path)
        print("Directory '% s' created" % pdfDirName)

    if os.path.exists(csvDirName):
        print("directory already exits")
        #shutil.rmtree(csvDirName)
    else:
        path = os.path.join(parent_dir, csvDirName)
        os.mkdir(path)
        print("Directory '% s' created" % csvDirName)
    
    if os.path.exists(txtDirName):
        print("directory already exits")
        #shutil.rmtree(txtDirName)
    else:
        path = os.path.join(parent_dir, txtDirName)
        os.mkdir(path)
        print("Directory '% s' created" % txtDirName)

    if os.path.exists(ldaDirName):
        print("directory already exits")
        #shutil.rmtree(txtDirName)
    else:
        path = os.path.join(parent_dir, ldaDirName)
        os.mkdir(path)
        print("Directory '% s' created" % ldaDirName)

    return pdfDirName, csvDirName, txtDirName, ldaDirName

def getArticleCount(url):
    page = requests.get(url)
    doc = BeautifulSoup(page.text, "html.parser")
    c = doc.find('script', id="serverApp-state").text
    result_list = re.findall('&q;/(.+?)&q;', c)
    number_of_articles_found = int(re.findall('PUBLIC&q;,&q;doc_count&q;:(.+?)}]', c)[0])
    return number_of_articles_found
    
def getPDFNameFromGivenURL(url):
    page = requests.get(url)
    doc = BeautifulSoup(page.text, "html.parser")
    c = doc.find('script', id="serverApp-state").text
    result_list = re.findall('&q;/(.+?)&q;', c)
    res = []
    [res.append(x) for x in result_list if x not in res and 'pdf' in x and 'txt' not in x and 'xlsx' not in x and 'pptx' not in x and 'docx' not in x and 'mp4' not in x and len(x) > 30]
    return res

def getDPFLink(result_list):
    pdf_link_list = []
    link_list = []
    for x in result_list:
        pdf_link_list.append(f"https://ntrs.nasa.gov/{x}")
        print(x)
        link = x.split("/", 1)[1]
        link = link.split("/downloads")[0]
        link_list.append(f"https://ntrs.nasa.gov/{link}")
    return pdf_link_list, link_list


def getAllPDFLinksFromSite(userInputTopic):

    #Convert user input topic into NASA websit's user input topic format
    userInputTopic = userInputTopic.replace("'", "")
    print(f"topic:", userInputTopic )
    input_topic_split = userInputTopic.split()
    key_words=""
    for word in input_topic_split:
        key_words+=word
        key_words+="%20"
    #remove last three charater
    topic_words = key_words[:-3]
    url = f'https://ntrs.nasa.gov/search?q="{topic_words}"'
    date = f'&published=%7B"gte":"2022-01-01"%7D'
    print(url)
    number_of_articles_found = getArticleCount(url+date)
    print(f"number_of_articles_found: {number_of_articles_found}")

    all_pdf_link_list = []
    all_link_list = []
    all_pdf_path = []
    list = []
    counter = 0

    for i in range (0, number_of_articles_found, 100):
        print(f"i: {i}")
        time.sleep(3)
        link = url + f'&page=%7B"size":100,"from":{i}%7D'
        print(link)
        result_list = getPDFNameFromGivenURL(link)
        list = list + result_list
        pdf_link_list, link_list = getDPFLink(result_list)

        for x in pdf_link_list:
            all_pdf_link_list.append(x)
        for x in link_list:
            all_link_list.append(x)

    return all_pdf_link_list, all_link_list

def downloadPDF(pdf_link, pdfDirName):

    # pdf_link_list = []
    # for i in pdf_link:
    #     pdf_link_list.append(i)

    pdf_name = []
    for file in pdf_link:
        time.sleep(1)
        response = ""
        download = False
        while (download == False):
            try:
                print('Sending request')
                response = requests.get(file, timeout=20)
                download = True
                print('data recived')
            except:
                print('Request has timed out, retrying')
        name = file.split("/")
        name = name[len(name)-1]
        path = f"{pdfDirName}/{name}"
        open(path, "wb").write(response.content)
        print(f'{name} downloaded')
        pdf_name.append(path)

    return pdf_name
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------



# Converting PDF into text format
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
def cleanText(text):
    text = text.lower()
    #remove references
    pos = text.rfind('references')
    text = text[:pos]
    text = text.replace(r'-', '')
    text = text.replace(r'_', '')
    text = text.replace(r')', '')
    text = text.replace(r'(', '')
    text = text.replace(r':', '')
    text = text.replace(r'?', '')
    text = text.replace(r'%', '')
    text = text.replace(r'+', '')
    text = text.replace(r'=', '')
    text = text.replace(r'&', '')
    text = text.replace(r'^', '')
    text = text.replace(r'$', '')
    text = text.replace(r'#', '')
    text = text.replace(r'!', '')
    text = text.replace(r'~', '')
    text = text.replace(r'`', '')
    text = text.replace(r'[', '')
    text = text.replace(r']', '')
    text = text.replace(r'{', '')
    text = text.replace(r'}', '')
    text = text.replace(r'\\', '')
    text = text.replace(r'|', '')
    text = text.replace(r';', '')
    text = text.replace(r'<', '')
    text = text.replace(r'>', '')
    text = text.replace(r'"', '')
    text = text.replace(r'*', '')
    text = text.replace(r'/', '')
    
    text = re.sub(r'\S*@\S*\s?', '', text)  # remove emails
    text = re.sub(r'\s+', ' ', text)  # remove newline chars
    text = re.sub(r"\'", "", text)  # remove single quotes
    text = re.sub(r"http[s]?\://\S+","",text) #removing http:
    text = re.sub(r"[0-9]", "",text) # removing number
    text = re.sub(r'\s+',' ',text) # removing space
    text = text.encode('ascii', 'ignore').decode()
    
    return text

def extractTextFromScannedPDFs(fname):
    text = ""
    while True:
        #code to extract text from scanned pdfs
        pages = convert_from_path(fname)
        try:
            for page in pages:
                text += pytesseract.image_to_string(page)
            new_text = text.strip("\n")
        except:
            print("Oops!  PDF not readable")
            break
    return new_text


def extractTextFromPDFs(fname):
    #code to extract text from pdfs

    text = ""
    while True:
        try:
            pdfFileObj = open(fname, 'rb')
            pdfReader = PyPDF2.PdfReader(pdfFileObj,strict=False)
            text = ""
            print(f'page{pdfReader.numPages}')
            if (pdfReader.numPages > 2):
                for i in range(pdfReader.numPages):
                    pageObj = pdfReader.getPage(i)
                    text += pageObj.extractText()
                pdfFileObj.close()
                break
            break
        except:
            print("Oops!  PDF not readable")
            break

    return text

def getText(path):
    text = extractTextFromPDFs(path)
    print("text length:" )
    print(len(text))
    if (len(text) < 3000):
        #print("Scanned PDF")
        text = extractTextFromScannedPDFs(path)
    return text
    
def convertPDF(pdf_name, txtDirName):

    txt_name = []
    for pdf in pdf_name:
        text = getText(pdf)
        text = cleanText(text)
        text_name = pdf.split('/')
        text_name = text_name[1].split('.')
        text_name = f"{txtDirName}/{text_name[0]}.txt"
        with open(text_name, 'w') as f:
            f.write(text)
            f.close() 
        txt_name.append(text_name)       

    return txt_name
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


# LDA Topic Modeling 
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# data cleaning
def sentences_to_words(sentences):
    for sent in sentences:
        #convert sentence to word
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

# remove stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# lemmatized words into their root form
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def getLDAModel(textfile, ldaDirName, pdf_link):
    data = []
    data.append(textfile)
    sentences = data
    data_words = list(sentences_to_words(sentences))

    bigram = gensim.models.Phrases(data_words, min_count=1, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    # Build LDA model
    if (len(data_lemmatized[0]) > 5):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=1, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)
        lda_name = pdf_link.split('/')
        lda_name = lda_name[len(lda_name)-1]
        lda_name = f"{ldaDirName}/{lda_name}.html"
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
        pyLDAvis.save_html(vis,lda_name)
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
        # a measure of how good the model is. lower the better.
        return lda_model, lda_name
    else:
        return None, ''

def getTopicKeywordsWithItsWeight(ldaModel):

    topic_list = ldaModel.show_topic(0, topn=30)
    res = dict()
    for x in topic_list:
        res[x[0]] = x[1]
    return res

def lda_model_listgetWeightList(new_pdf_link, new_link, new_txt_name, new_lda_topic_key_word_list, lda_model_list, userInputTopic, lda_dir_name):


    userInputTopicKeyWord = userInputTopic.split()
    print(f"userInputTopicKeyWord: {userInputTopicKeyWord}")
    weight_list = []
    for y in range(0, len(lda_model_list)):
        keywordWithItsWeightList = getTopicKeywordsWithItsWeight(lda_model_list[y])
        print(keywordWithItsWeightList)

        weight = 0
        for x in userInputTopicKeyWord:
            result = keywordWithItsWeightList.get(x)
            if result == None:
                result = 0
            weight += result
        weight_list.append(weight)

    data = {'pdf_link': new_pdf_link, 'link': new_link, 'txt_name': new_txt_name, 'new_lda_topic_key_word_list' : new_lda_topic_key_word_list, 'lda_dir_name' : lda_dir_name, 'weight' : weight_list}
    df = pd.DataFrame(data,columns=['pdf_link', 'link', 'txt_name', 'new_lda_topic_key_word_list', 'lda_dir_name', 'weight'])
    new_list = df.sort_values(by=['weight'], ascending=False)
    return new_list
    


def getTopic(all_files, userInputTopic, ldaDirName):

    text_content_list = []
    for txt in all_files.txt_name:
        text = open(txt,'r').read()
        text_content_list.append(text)

    new_pdf_link = []
    new_link = []
    new_text = []
    lda_model_list = []
    new_lda_topic_key_word_list = []
    lda_dir_name = []
    n = 0
    for x in text_content_list:
        print(len(x))
        if (len(x) > 200 and len(x) < 1000000):
            lda_model, lda_name = getLDAModel(x, ldaDirName, all_files.pdf_link[n])
            if (lda_model != None):
                print(n)
                new_pdf_link.append(all_files.pdf_link[n])
                new_link.append(all_files.link[n])
                new_text.append(all_files.txt_name[n])
                lda_model_list.append(lda_model)
                lda_dir_name.append(lda_name)
                topic = lda_model.show_topic(0, topn=30)
                p = ''
                for i in topic:
                    p = p + i[0] + "+" + str(i[1]) + " "
                    
                print(p)
                print(topic)
                new_lda_topic_key_word_list.append(p)
                print("added")
        else:
            print(f"{all_files.txt_name} is too large")
        n+=1
    
    new_list = lda_model_listgetWeightList(new_pdf_link, new_link, new_text, new_lda_topic_key_word_list,lda_model_list, userInputTopic, lda_dir_name)
    
    return new_list, text_content_list
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
def getListWithWeightedKeyWord(weight_pdf_list):
    new_pdf_link_list = []
    new_link_list = []
    new_topic_key_word = []
    new_weight_list = []
    new_text_name = []
    count = 0
    for i in weight_pdf_list.weight:
        if (i > 0):
            new_pdf_link_list.append(weight_pdf_list.pdf_link[count])
            new_link_list.append(weight_pdf_list.link[count])
            new_topic_key_word.append(weight_pdf_list.new_lda_topic_key_word_list[count])
            new_weight_list.append(weight_pdf_list.weight[count])
            new_text_name.append(weight_pdf_list.txt_name[count])
        count+=1

    new_list = {'pdf_link': new_pdf_link_list, 'link': new_link_list, 'txt_name' : new_text_name, 'new_lda_topic_key_word_list' : new_topic_key_word, 'weight' : new_weight_list}
    new_list = pd.DataFrame(new_list)

    return new_list

def compareLDAModel(ldaModel1, ldaModel2):
    topic_list1 = ldaModel1.split()
    topic_list2 = ldaModel2.split()

    # print(f'topic1: {topic_list1}')
    # print(f'topic2: {topic_list2}')

    key_word_list1 = []
    key_word_list2 = []
    for i in topic_list1:
        temp = i.split("+")
        key_word_list1.append(temp[0])

    for i in topic_list2:
        temp = i.split("+")
        key_word_list2.append(temp[0])

    # print(f'topic1: {key_word_list1}')
    # print(f'topic2: {key_word_list2}')

    count = 0
    for x in key_word_list1:
        for y in key_word_list2:
            if (x == y):
                count +=1
    return count

def getSimilarArticleList(data):
    link_list = data.pdf_link.values.tolist()
    key_word_list = data.new_lda_topic_key_word_list.tolist()
    txtdir = data.txt_name.tolist()

    similar_article_list1 = [] 
    similar_article_list2 = [] 
    topic_list1 = []
    topic_list2 = []
    similarity = []

    for i in range (0, len(key_word_list), 1):
        for j in range (0, len(key_word_list), 1):
            if(link_list[i] != link_list[j]):
                if (compareLDAModel(key_word_list[i], key_word_list[j]) >=15):
                    similar_article_list1.append(link_list[i])
                    similar_article_list2.append(link_list[j])
                    topic_list1.append(key_word_list[i])
                    topic_list2.append(key_word_list[j])
                    similarity.append(getDec2Vec(txtdir[i], txtdir[j]))
            
            j +=1 
        similar_article_list1.append('')
        similar_article_list2.append('')
        topic_list1.append('')
        topic_list2.append('')
        similarity.append('')
        i +=1


    return similar_article_list1, similar_article_list2, topic_list1, topic_list2, similarity
# Doc2Vec
def get_taggeddoc(txt_list):
    
    #read all the documents under given folder name
    doc_list = txt_list
    
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
        #print(tokens)
 
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        #print(stopped_tokens)
        #print("--------------------------------------------------------------")
        # remove numbers
        number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        number_tokens = ' '.join(number_tokens).split()

        #print(number_tokens)
        #print("--------------------------------------------------------------")
        # stem tokens
        lemmatized_tokens = [lemmatizer.lemmatize(i) for i in number_tokens]
        #print(lemmatized_tokens)
        #print("--------------------------------------------------------------")
        
        # remove empty
        length_tokens = [i for i in lemmatized_tokens if len(i) > 1]
        # add tokens to list
        lemmatized_tokens.append(",")

        texts.append(lemmatized_tokens)
        
 
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(lemmatized_tokens))).split(),str(index))
        taggeddoc.append(td)
        #print(taggeddoc)
 
    return taggeddoc

def getDec2Vec(article1, article2):
    # build the model
    #model =  gensim.models.doc2vec.Doc2Vec(taggeddoc, vector_size=30, min_count=1, epochs=80)
    txt_list = []
    txt_list.append(open(article1,'r').read())
    txt_list.append(open(article2,'r').read())

    taggeddoc = get_taggeddoc(txt_list)


    model =  gensim.models.doc2vec.Doc2Vec(taggeddoc, vector_size=30, min_count=1, epochs=80)
    #model.build_vocab(taggeddoc)
    #model.train(taggeddoc, total_examples=model.corpus_count, epochs=model.epochs)
    return model.dv.similarity(0, 1)
    
# Main method
# -----------------------------------------------------------------------------------------------------
def search(userInputTopic, lang="en"):
    
    # # # Create new directory to save the files for given topic
    pdfDirName, csvDirName, txtDirName, ldaDirName= makeNewDiretoryForGivenTopic(userInputTopic)
    tablename = userInputTopic.replace(" ", "")
    # # #Get all the pdf links from NASA website given usuer input topic
    # # # #----------------------------------------------------------------------------------------------------
    # all_pdf_link_list, all_link_list = getAllPDFLinksFromSite(userInputTopic)
    # #Save it into csv file
    # paper = {'pdf_link': all_pdf_link_list, 'link': all_link_list}
    # papers = pd.DataFrame(paper)
    # print(papers)
    # papers.to_csv(f'{csvDirName}/all1.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # # # #----------------------------------------------------------------------------------------------------

    # # # # # Downloading all the files
    # # # # #----------------------------------------------------------------------------------------------------
    # # Read the links from the csv file
    # links = pd.read_csv(f'{csvDirName}/all1.csv')
    # pdf_name = downloadPDF(links.pdf_link, pdfDirName)
    # links['pdf_name'] = pdf_name
    # links.to_csv(f'{csvDirName}/all2.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # # #----------------------------------------------------------------------------------------------------

    # # # Converting PDFs into text format
    # # #----------------------------------------------------------------------------------------------------
    # pdfName = pd.read_csv(f'{csvDirName}/all2.csv')
    # txt_name = convertPDF(pdfName.pdf_name, txtDirName)
    # pdfName['txt_name'] = txt_name
    # pdfName.to_csv(f'{csvDirName}/all3.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # # #-----------------------------------------------------------------------------------------------------


    # # Finding topic key words for each articles
    # #----------------------------------------------------------------------------------------------------
    # all_files = pd.read_csv(f'{csvDirName}/all3.csv')
    # # new_pdf_link, new_link, new_lda_topic_key_word_list, lda_model_list = 
    # new_list,text_content_list = getTopic(all_files, userInputTopic, ldaDirName)
    # new_list.to_csv(f'{csvDirName}/all4.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # all_text_content_list = {'text': text_content_list,}
    # text_content_list = pd.DataFrame(all_text_content_list)
    # text_content_list.to_csv(f'{csvDirName}/txt.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # #-----------------------------------------------------------------------------------------------------

    # # Listing articles by their Weighted key word
    # #----------------------------------------------------------------------------------------------------   
    # weight_pdf_list = pd.read_csv(f'{csvDirName}/all4.csv')
    # new_link_list = getListWithWeightedKeyWord(weight_pdf_list)
    # new_link_list.to_csv(f'{csvDirName}/all5.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # HTML(new_link_list.to_html(f'table/{tablename}all5.html', render_links=True, escape=False)) 
    # #-----------------------------------------------------------------------------------------------------

    # # Getting Similarities between articles 
    # # #-----------------------------------------------------------------------------------------------------
    # data = pd.read_csv(f'{csvDirName}/all5.csv')
    # similar_article_list1, similar_article_list2, topic_list1, topic_list2, similarity = getSimilarArticleList(data)
    # new_list = {'similar_article_list1': similar_article_list1, 'similar_article_list2' : similar_article_list2, 'topic_list1': topic_list1, 'topic_list2' : topic_list2, 'similarity' : similarity}
    # new_list = pd.DataFrame(new_list)
    # new_list.to_csv(f'{csvDirName}/all6.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # HTML(new_list.to_html(f'table/{tablename}all6.html', render_links=True, escape=False)) 

    return_list  = []
    return_list.append(f'table/{tablename}all5.html')
    return_list.append(f'table/{tablename}all6.html')

    return return_list
    
