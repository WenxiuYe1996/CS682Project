from bs4 import BeautifulSoup
import requests
import re
import time
from requests.exceptions import ConnectTimeout


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
    #date = f'&published=%7B"gte":"2020-01-01"%7D'
    print(url)
    number_of_articles_found = getArticleCount(url)
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
    
all_pdf_link_list, all_link_list = getAllPDFLinksFromSite('climate change')
for i in all_pdf_link_list:
    print(i)


