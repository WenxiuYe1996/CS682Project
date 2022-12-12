import PyPDF2
import io
import re
import pytesseract
from pdf2image import convert_from_path

#extract text from scanned pdf
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

#extract text from generated pdf
def extractTextFromPDFs(fname):
    text = ""
    while True:
        try:
            pdfFileObj = open(fname, 'rb')
            pdfReader = PyPDF2.PdfReader(pdfFileObj,strict=False)
            text = ""
            print(f'page{pdfReader.numPages}')
            for i in range(pdfReader.numPages):
                pageObj = pdfReader.getPage(i)
                text += pageObj.extractText()
            pdfFileObj.close()
            break
        except:
            print("Oops!  PDF not readable")
            break

    return text

#clean the text
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


def getText(path):
    text = extractTextFromPDFs(path)
    print("text length:" )
    print(len(text))
    if (len(text) < 5):
        text = extractTextFromScannedPDFs(path)
    return text

    
text = getText(f'b.pdf')
print(text)

with open(f'b.txt', 'w') as f:
        f.write(text)
        f.close() 
