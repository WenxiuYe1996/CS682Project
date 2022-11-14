# from urllib import request
from flask import Flask, render_template, url_for, request
from bs4 import BeautifulSoup
import requests
import re

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

text = ""

@app.route('/result', methods =["GET", "POST"])
def text_extract():
    if request.method == "POST":
       # getting input 
       text = request.form.get("ip")

       url = f"https://ntrs.nasa.gov/search?q={text}"
       #print(url)

       page = requests.get(url)
       doc = BeautifulSoup(page.text, "html.parser")

       result_list = re.findall(':&q;/api/citations/(.+?)/downloads/', doc.find(text=re.compile('citations')))
       
       links = []
       [links.append(x) for x in result_list if x not in links]

       #print(links)

       link_list= []
       pdf_link_list= []
       for r in links:
          link_list.append(f"https://ntrs.nasa.gov/citations/{r}")
          pdf_link_list.append(f"https://ntrs.nasa.gov/api/citations/{r}/downloads/")
          #print(link_list)

       c = doc.find('script', id="serverApp-state").text
       result_list = re.findall('/downloads/(.+?)&q', c)
       res = []
       [res.append(x) for x in result_list if x not in res and  'txt' not in x]
       print(res)
    return render_template("result.html", result = res, link = link_list, pdf_link = pdf_link_list)



if __name__ == "__main__":
    app.run(debug=True)