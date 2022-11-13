from bs4 import BeautifulSoup
import requests
import re
import pprint

#get user input topic
   
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
[res.append(x) for x in result_list if x not in res and 'txt' not in x and 'xlsx' not in x and 'api' in x]


pdf_link_list = []
link_list = []
print(res)
for x in res:
    pdf_link_list.append(f"https://ntrs.nasa.gov/{x}")
    print(x)
    link = x.split("/", 1)[1]
    link = link.split("/downloads")[0]
    link_list.append(f"https://ntrs.nasa.gov/{link}")
    
print(f"length of pdf: {len(pdf_link_list)}")
for x in pdf_link_list:
    print(x)

print(f"length of link_list: {len(link_list)}")
for x in link_list:
    print(x)


