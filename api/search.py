from bs4 import BeautifulSoup
import requests
import re
import csv
import shutil
import os
import time
import pandas as pd
from IPython.display import HTML

# Create new directories for given topic
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
def makeNewDiretoryForGivenTopic(userInputTopic):
    
    dirName = userInputTopic.replace(" ", "")

    parent_dir = dirName

    if os.path.exists(parent_dir):
        print("directory already exits")
        #shutil.rmtree(pdfDirName)
    else:
        path = os.path.join("", parent_dir)
        os.mkdir(path)
        print("Directory '% s' created" % parent_dir)

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
    

    for dir_name in directory_list:
        # Check if directory already existed
        if os.path.exists(f'{parent_dir}/{dir_name}'):
            print("directory already exits")
            #shutil.rmtree(pdfDirName)
        # Create directory if not exist
        else:
            print(f'{parent_dir}/dir_name')
            path = os.path.join(parent_dir, dir_name)
            os.mkdir(path)
            print("Directory '% s' created" % dir_name)

    pdfDirName = f"{parent_dir}/pdf{dirName}"
    paperDirName = f"{parent_dir}/paper{dirName}"
    imageDirName = f"{parent_dir}/image{dirName}"
    csvDirName = f"{parent_dir}/csv{dirName}"
    txtDirName = f"{parent_dir}/txt{dirName}"
    ldaDirName = f"{parent_dir}/lda{dirName}"

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

    # Convert user input topic into NASA websit's user input topic format
    userInputTopic = userInputTopic.replace("'", "")
    print(f"topic:", userInputTopic )
    input_topic_split = userInputTopic.split()
    key_words=""
    for word in input_topic_split:
        key_words += word
        key_words += "%20"
    # Remove last three charater
    topic_words = key_words[:-3]
    url = f'https://ntrs.nasa.gov/search?q="{topic_words}"'
    # Set a time frame 
    date = f'&published=%7B"gte":"2022-03-01"%7D'

    url = url + date

    print(url)
    number_of_articles_found = getArticleCount(url)

    print(f"number_of_articles_found: {number_of_articles_found}")

    all_pdf_link_list = []
    all_link_list = []
    list = []

    # For every result pages
    # Get all the links to articles and links to the description of the articles
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

# Download the files found on NASA website
def downloadPDF(pdf_link, pdfDirName):
    pdf_name_list = []
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
        pdf_name_list.append(path)

    return pdf_name_list
###############################################################################################################
###############################################################################################################


# Main method
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def search(userInputTopic):
    
    # # ----------------------------------------------------------------------------------------------------
    # Create new directory to save the files for given topic
    pdfDirName, paperDirName, imageDirName, csvDirName, txtDirName, ldaDirName = makeNewDiretoryForGivenTopic(userInputTopic)
    
    #  ----------------------------------------------------------------------------------------------------
    #  Get all the resources from NASA website given user input topic
    all_pdf_link_list, all_link_list = getAllPDFLinksFromSite(userInputTopic)
    # Save the links to the articles into a csv file
    paper_link = {'pdf_link': all_pdf_link_list, 'link': all_link_list}
    paper_links = pd.DataFrame(paper_link)
    paper_links_csv = f'{csvDirName}/paper_links.csv'
    paper_links.to_csv(paper_links_csv, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # Generate a table for paper_links in a HTML file
    paper_links_html = f'{csvDirName}/paper_links.html'
    HTML(paper_links.to_html(paper_links_html, render_links=True, escape=False)) 

    # # ----------------------------------------------------------------------------------------------------
    # Download all the files from NASA Website 
    pdf_path_list = downloadPDF(all_pdf_link_list, pdfDirName)
    paper_link_path = {'pdf_link': all_pdf_link_list, 'link': all_link_list, 'path': pdf_path_list}
    paper_link_paths = pd.DataFrame(paper_link_path)
    # Save the information in a csv file
    paper_link_path_dir = f'{csvDirName}/paper_link_path.csv'
    paper_link_paths.to_csv(paper_link_path_dir, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # Make a html version of the csv file
    paper_link_path_html = f'{csvDirName}/paper_link_path.html'
    HTML(paper_link_paths.to_html(paper_link_path_html, render_links=True, escape=False)) 

    # # ----------------------------------------------------------------------------------------------------
    # Get the total number of papers downloaded
    total_num_paper = len(pdf_path_list)
    print(f'Number of paper Downloaded: {total_num_paper}')

    # Append total number of papers downloaded to a list
    total_num_paper_list  = []
    total_num_paper_list.append(f'Number of paper Downloaded: {total_num_paper}')

    # Append the link to the html to a list
    table_name_list = []
    table_name_list.append(paper_link_path_html)

    return total_num_paper_list, table_name_list


###############################################################################################################
###############################################################################################################