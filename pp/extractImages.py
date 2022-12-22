from bs4 import BeautifulSoup
import requests
import re
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import PyPDF2
import csv
import fitz
import time
import shutil
from IPython.display import HTML
from requests.exceptions import ConnectTimeout
import urllib.request
import PyPDF2
import io
import csv
import pytesseract
from pdf2image import convert_from_path
import shutil

import re
import os
import pandas as pd
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

#Create new directory for given topic
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
def makeNewDiretoryForGivenTopic(userInputTopic):
    
    dirName = userInputTopic.replace(" ", "")

    parent_dir = ""
    pdfDirName = f"pdf{dirName}"
    paperDirName = f"paper{dirName}"
    imageDirName = f"image{dirName}"
    csvDirName = f"csv{dirName}"
    txtDirName = f"txt{dirName}"
    ldaDirName = f"lda{dirName}"

    directory_list = []
    directory_list.append(pdfDirName)
    directory_list.append(paperDirName)
    directory_list.append(imageDirName)
    directory_list.append(csvDirName)
    directory_list.append(txtDirName)
    directory_list.append(ldaDirName)
    
    # Create New Directories for given topic
    for dir_name in directory_list:
        if os.path.exists(dir_name):
            print("directory already exits")
            #shutil.rmtree(pdfDirName)
        else:
            path = os.path.join(parent_dir, dir_name)
            os.mkdir(path)
            print("Directory '% s' created" % dir_name)


    return pdfDirName, paperDirName, imageDirName, csvDirName, txtDirName, ldaDirName
###############################################################################################################
###############################################################################################################



# Web Scraper for NASA website
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
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
###############################################################################################################
###############################################################################################################


# Converting PDF into text format
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
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
    # if (len(text) < 3000):
    #     #print("Scanned PDF")
    #     text = extractTextFromScannedPDFs(path)
    return text

###############################################################################################################
###############################################################################################################



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

def getLDAModel(textfile, ldaDirName, pdf_path):
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
        lda_name = pdf_path.split('/')
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
###############################################################################################################
###############################################################################################################


def getTopicKeywordsWithItsWeight(ldaModel):

    topic_list = ldaModel.show_topic(0, topn=30)
    res = dict()
    for x in topic_list:
        res[x[0]] = x[1]
    return res

def lda_model_listgetWeightList(new_pdf_path_list, new_text_path, new_lda_topic_key_word_list,lda_model_list, userInputTopic, lda_dir_name, new_image_list):


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

    data = {'pdf_path': new_pdf_path_list, 'text_path': new_text_path, 'lda_topic_key_word_list' : new_lda_topic_key_word_list, 'lda_dir_name' : lda_dir_name, 'weight' : weight_list, 'images' : new_image_list}
    df = pd.DataFrame(data,columns=['pdf_path', 'text_path', 'lda_topic_key_word_list', 'lda_dir_name', 'weight', 'images'])
    new_list = df.sort_values(by=['weight'], ascending=False)
    return new_list
    


def getTopic(all_files, userInputTopic, ldaDirName):

    # Get the content of each papars and store them using a list
    text_content_list = []
    for txt in all_files.text_path:
        text = open(txt,'r').read()
        text_content_list.append(text)

    new_pdf_path_list = []
    lda_model_list = []
    new_lda_topic_key_word_list = []
    lda_dir_name = []
    new_image_list = []
    new_txt_path_list = []
    n = 0
    min_text = 200
    max_text = 1000000
    # Get topic for each paper
    for text in text_content_list:
        print(len(text))

        if (len(text) > min_text and len(text) < max_text):

            lda_model, lda_name = getLDAModel(text, ldaDirName, all_files.paper_path[n])
            if (lda_model != None):
                print(n)
                new_pdf_path_list.append(all_files.paper_path[n])
                lda_model_list.append(lda_model)
                lda_dir_name.append(lda_name)
                new_image_list.append(all_files.image_path[n])
                new_txt_path_list.append(all_files.text_path[n])
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
    
    new_list = lda_model_listgetWeightList(new_pdf_path_list, new_txt_path_list, new_lda_topic_key_word_list,lda_model_list, userInputTopic, lda_dir_name, new_image_list)
    
    return new_list, text_content_list
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
def getListWithWeightedKeyWord(weight_pdf_list):
    new_pdf_path_list = []
    new_topic_key_word = []
    new_weight_list = []
    new_text_path = []
    new_image_path = []
    count = 0
    for i in weight_pdf_list.weight:
        if (i > 0):
            new_pdf_path_list.append(f'http://127.0.0.1:5000/{weight_pdf_list.pdf_path[count]}')
            new_topic_key_word.append(weight_pdf_list.lda_topic_key_word_list[count])
            new_weight_list.append(weight_pdf_list.weight[count])
            new_text_path.append(f'http://127.0.0.1:5000/{weight_pdf_list.text_path[count]}')
            new_image_path.append(f'http://127.0.0.1:5000/{weight_pdf_list.images[count]}')
        count+=1

    new_list = {'pdf_path': new_pdf_path_list,'text_path' : new_text_path, 'lda_topic_key_word_list' : new_topic_key_word, 'weight' : new_weight_list, 'images' : new_image_path}
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
    link_list = data.pdf_path.values.tolist()
    key_word_list = data.lda_topic_key_word_list.tolist()
    txtdir = data.text_path.tolist()
    images_path = data.images.tolist()

    similar_article_list1 = [] 
    similar_article_list2 = [] 
    topic_list1 = []
    topic_list2 = []
    similarity = []
    images_path_list = []

    for i in range (0, len(key_word_list), 1):
        for j in range (0, len(key_word_list), 1):
            if(link_list[i] != link_list[j]):
                if (compareLDAModel(key_word_list[i], key_word_list[j]) >=15):
                    similar_article_list1.append(link_list[i])
                    similar_article_list2.append(link_list[j])
                    topic_list1.append(key_word_list[i])
                    topic_list2.append(key_word_list[j])
                    images_path_list.append(f'{images_path[i]} , {images_path[j]}')
                    similarity.append(getDec2Vec(txtdir[i], txtdir[j]))
            
            j +=1 
        similar_article_list1.append('')
        similar_article_list2.append('')
        topic_list1.append('')
        topic_list2.append('')
        similarity.append('')
        images_path_list.append('')
        i +=1


    return similar_article_list1, similar_article_list2, topic_list1, topic_list2, images_path_list, similarity
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

def saveImagesIntoPdf(file_path, images_path, error):


    try:
        print(f'images_path:{images_path}')
        doc = fitz.open()  # PDF with the pictures
        imgdir = "img"  # where the pics are
        imglist = os.listdir(imgdir)  # list of them
        imgcount = len(imglist)  # pic count

        for i, f in enumerate(imglist):
            img = fitz.open(os.path.join(imgdir, f))  # open pic as document
            rect = img[0].rect  # pic dimension
            pdfbytes = img.convert_to_pdf()  # make a PDF stream
            img.close()  # no longer needed
            imgPDF = fitz.open("pdf", pdfbytes)  # open stream as PDF
            page = doc.new_page(width = rect.width,  # new page with ...
                                    height = rect.height)  # pic dimension
            page.show_pdf_page(rect, imgPDF, 0)  # image fills the page
                    

        doc.save(images_path)
    except:
        error = True
        print("Error occurred when Saving imagges into one pdf file")
        
    return error

def getImagesFromPDF(file_path, images_path):
    imgdir = "img"
    error = False

    if os.path.exists(imgdir):
        shutil.rmtree(imgdir)
        path = os.path.join("", imgdir)
        os.mkdir(path)
        print("Directory img created")
    else:
        path = os.path.join("", imgdir)
        os.mkdir(path)
        print("Directory img created")

    # Extract Images from current file and save them under directory img
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
      # where the pics are
    try:
    
        doc = fitz.open(file_path)
        imagelist = []
        print(type(doc))
        for i in range(len(doc)):
            for img in doc.get_page_images(i):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:       # this is GRAY or RGB
                    pix.save(f"{imgdir}/p%s-%s.png" % (i, xref))
                else:               # CMYK: convert to RGB first
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    imagelist.append(pix1)
                    pix1.save(f"{imgdir}/p%s-%s.png" % (i, xref))
                    pix1 = None
                pix = None
    except:
        error = True
        print("Error occurred when extracting imagges")
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    error = saveImagesIntoPdf(file_path, images_path, error)

    # Save all the images extracted into one pdf file
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    return error

def extractImagesFromFile(paperList_textList, pdfDirName, imageDirName):
    paperList = paperList_textList.paper_path
    images_path_list = []
    # Extract Images from the paper
    # ---------------------------------------------------------------------------
    for paper in paperList:
        images_path =  paper.replace(pdfDirName, imageDirName)
        print(f"images_path: {images_path}")
        if (getImagesFromPDF(paper, images_path) == False):
            print("Successuflly extracted images")
        else:
            print("Failed to extract image")
        images_path_list.append(images_path)
    # ---------------------------------------------------------------------------

    return images_path_list



    
def extractPaperInfo(files, pdfDirName, paperDirName, txtDirName):

    paper_path_list = []
    text_file_list = []
    #images_path_list =[]
    min_page = 5
    max_page = 50

    for file in files:

        try:
            pdfFileObj = open(file, 'rb')
            pdfReader = PyPDF2.PdfReader(pdfFileObj,strict=False)

            # Check the number of pages for this file
            if (pdfReader.numPages > min_page and pdfReader.numPages < max_page):

                # Extract Text from the file
                text = getText(file)
                # Clean the text by removing punctuations, quotation mark, numbers...etc
                text = cleanText(text)

                # Find the position of abstract
                abstract_pos = text.rfind('abstract')
                # Find the position of references
                references_pos = text.rfind('references')

                # if the article contains abstract and reference, count it as a paper
                if (references_pos != -1):

                    # Make a copy of this paper and save it under folder {paperDirName}
                    # ---------------------------------------------------------------------------
                    shutil.copy(file, paperDirName)
                    paper_path =  file.replace(pdfDirName, paperDirName)
                    paper_path_list.append(paper_path)
                    # ---------------------------------------------------------------------------

                    # Save the text as text document
                    # ---------------------------------------------------------------------------
                    text_file_name = file.split('/')
                    text_file_name = text_file_name[1].split('.')
                    text_file_name = f"{txtDirName}/{text_file_name[0]}.txt"
                    with open(text_file_name, 'w') as f:
                        f.write(text)
                        f.close() 
                    text_file_list.append(text_file_name)
                    # ---------------------------------------------------------------------------

        except:
                print('Error occurred')

    return paper_path_list, text_file_list  #, images_path_list

# Main method
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def search(userInputTopic, lang="en"):
    

    ############################### Downloading Data Seciton ##############################################
    # # # Create new directory to save the files for given topic
    # # ----------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------
    pdfDirName, paperDirName, imageDirName, csvDirName, txtDirName, ldaDirName = makeNewDiretoryForGivenTopic(userInputTopic)
    topic = userInputTopic.replace(" ", "")
    # #
    # # # Get all the resources from NASA website given usuer input topic
    # # ----------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------
    # all_pdf_link_list, all_link_list = getAllPDFLinksFromSite(userInputTopic)
    # # Save the links to the articles into csv file
    # paper = {'pdf_link': all_pdf_link_list, 'link': all_link_list}
    # papers = pd.DataFrame(paper)
    # print(papers)
    # papers.to_csv(f'{csvDirName}/all1.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # # ----------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------
    # #
    # # # Downloading all the files from the NASA Website 
    # # ----------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------
    # # Read the links to the articles from the csv file
    # links = pd.read_csv(f'{csvDirName}/all1.csv')
    # # Download the files
    # pdf_name = downloadPDF(links.pdf_link, pdfDirName)
    # links['pdf_name'] = pdf_name
    # Save the path to each file in a csv file
    # links.to_csv(f'{csvDirName}/all2.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # # #----------------------------------------------------------------------------------------------------
    ##########################################################################################################



    ################################# Data Processing Section ################################################
    # User can add articles into the folder name {pdfDirName}
    # ----------------------------------------------------------------------------------------------------
    # Get All the files under directory {pdfDirName}
    # files_path  = [pdfDirName + '/' +name for name in os.listdir(pdfDirName) if name.endswith('pdf')]
    # # 
    # # Extract research pappers from the files and save them under folder name {paperDirName}
    # # Convert them into text document and save them under folder name {txtDirName}
    # paper_path_list, text_path_list = extractPaperInfo(files_path, pdfDirName, paperDirName, txtDirName)
    # # 
    # # Create a dictionary
    # paperList_textList = {'paper_path': paper_path_list, 'text_path': text_path_list}
    # # Build pandas dataframe for dictionary
    # paperList_textList = pd.DataFrame(paperList_textList)
    # #  Save the data into csv file
    # paperList_textList.to_csv(f'{csvDirName}/all3.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # 
    # Extract Images for every papers and save them under folder name {imageDirName}
    paperList_textList = pd.read_csv(f'{csvDirName}/all3.csv')
    imageList = extractImagesFromFile(paperList_textList, pdfDirName, imageDirName)
    paperList_textList_imageList = {'paper_path': paperList_textList.paper_pathq, 'text_path': paperList_textList.text_path, 'image_path': imageList}
    # # #-----------------------------------------------------------------------------------------------------
    ###########################################################################################################


    ################################# LDA Topic Modeling ################################################
    # # # Finding topic key words for each articles
    # # #----------------------------------------------------------------------------------------------------
    # all_files = pd.read_csv(f'{csvDirName}/all3.csv')
    # new_list,text_content_list = getTopic(all_files, userInputTopic, ldaDirName)
    # new_list.to_csv(f'{csvDirName}/all4.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # # # #-----------------------------------------------------------------------------------------------------
    # # # #
    # # # # Listing articles by their Weighted key word
    # # # #----------------------------------------------------------------------------------------------------   
    # weight_pdf_list = pd.read_csv(f'{csvDirName}/all4.csv')
    # new_link_list = getListWithWeightedKeyWord(weight_pdf_list)
    # new_link_list.to_csv(f'{csvDirName}/all5.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # table_name1 = f'table/{topic}all5.html'
    # HTML(new_link_list.to_html(table_name1, render_links=True, escape=False)) 
    # # # #-----------------------------------------------------------------------------------------------------
    # ###########################################################################################################

    # # # # Getting Similarities between articles 
    # # # # #-----------------------------------------------------------------------------------------------------
    # data = pd.read_csv(f'{csvDirName}/all5.csv')
    # similar_article_list1, similar_article_list2, topic_list1, topic_list2, images_path_list, similarity = getSimilarArticleList(data)
    # new_list = {'similar_article_list1': similar_article_list1, 'similar_article_list2' : similar_article_list2, 'topic_list1': topic_list1, 'topic_list2' : topic_list2, 'images' : images_path_list,  'similarity' : similarity}
    # new_list = pd.DataFrame(new_list)
    # new_list.to_csv(f'{csvDirName}/all6.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # table_name2 = f'table/{topic}all6.html'
    # HTML(new_list.to_html(table_name2, render_links=True, escape=False)) 

    # total_num_paper = len(weight_pdf_list)
    # num_paper_left = len(new_link_list)
    return_list  = []
    # return_list.append(f'Number of paper processed: {total_num_paper}')
    # return_list.append(f'Number of paper left: {num_paper_left}')
    # return_list.append(f'table/{tablename}all6.html')
    table_name_list = []
    # table_name_list.append(table_name1)
    # table_name_list.append(table_name2)
    return return_list, table_name_list
    

###############################################################################################################
###############################################################################################################