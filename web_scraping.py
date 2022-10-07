from bs4 import BeautifulSoup
import requests
import re
import wget

#get user input topic
input_topic = input("What topic do you want to search for? ")

#url for nasa website
url = f"https://ntrs.nasa.gov/search?q={input_topic}"
print(url)

#use BeautifulSoup to get the html information
page = requests.get(url)
doc = BeautifulSoup(page.text, "html.parser")

#get all the link from the returned search result
result_list = re.findall(':&q;/api/citations/(.+?)/downloads/', doc.find(text=re.compile('citations')))

#remove duplicate
links = []
[links.append(x) for x in result_list if x not in links]

#get all the links of returned result
link_list= []
pdf_link_list= []
for r in links:
    link_list.append(f"https://ntrs.nasa.gov/citations/{r}")
    pdf_link_list.append(f"https://ntrs.nasa.gov/api/citations/{r}/downloads/")
    print(link_list)

#get all the articles of returned result
c = doc.find('script', id="serverApp-state").text
result_list = re.findall('/downloads/(.+?)&q', c)
res = []
[res.append(x) for x in result_list if x not in res and  'txt' not in x]

#get all the pdf
pdf_list= []

i = 0
for r in res:
    pdf_list.append(pdf_link_list[i]+r)
    i+=1
    
for p in pdf_list:
    print(p)

i = 0
#download the pdfs
for p in pdf_list:
    response = requests.get(p)
    open(f"pdf/{res[i]}", "wb").write(response.content)
    i+=1
    
