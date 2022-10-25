import PyPDF2


#code to extract text from pdfs
pdfFileObj = open('b.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

num_pages = pdfReader.numPages
print(pdfReader.numPages)

for i in range(num_pages):
    pageObj = pdfReader.getPage(i)
    print(pageObj.extractText())

pdfFileObj.close()