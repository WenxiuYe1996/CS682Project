import PyPDF2

# creating a pdf file object
pdfFileObj = open('b.pdf', 'rb')

# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# printing number of pages in pdf file

num_pages = pdfReader.numPages
print(pdfReader.numPages)

for i in range(num_pages):
    # creating a page object
    pageObj = pdfReader.getPage(i)
    # extracting text from page
    print(pageObj.extractText())
    # closing the pdf file object
pdfFileObj.close()