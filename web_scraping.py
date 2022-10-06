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
#print(doc.prettify())

#page_citation = doc.find("a", class_="capitalize-title ng-star-inserted")

#get all the link from the returned search result
c = doc.find(text=re.compile('citations'))
result_list = re.findall(':&q;/api/citations/(.+?)/downloads/', c)

#remove duplicate
#res = [*set(b)]
res = []
[res.append(x) for x in result_list if x not in res]

#get all the link to the subtitle
res_link_list= []
for r in res:
    res_link_list.append(f"https://ntrs.nasa.gov/citations/{r}")

for p in res_link_list:
    print(p)


#get all the pdf
res_pdf_list= []
link_url = f"https://ntrs.nasa.gov/citations/"
for r in res:
    res_pdf_list.append(f"https://ntrs.nasa.gov/api/citations/{r}/downloads/{r}.pdf")
    
for p in res_pdf_list:
    print(p)

i = 0
#download the pdfs
for p in res_pdf_list:
    response = requests.get(p)
    open(f"pdf/{res[i]}.pdf", "wb").write(response.content)
    i+=1

#response = wget.download(url, "pdf/{i}.pdf")
    
