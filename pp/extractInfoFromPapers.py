from PIL import Image
import fitz
import shutil
from IPython.display import HTML
import PyPDF2
import csv
import pytesseract
from pdf2image import convert_from_path
import re
import os
import pandas as pd


#Create new directory for given topic
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
    
    # Create New Directories for given topic
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


# Text Cleanning
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
def cleanText(text):
    # Cleanning text
    text = text.lower()
    pos = text.rfind('references')# Remove references
    text = text[:pos]
    text = text.replace(r'-', '')
    text = text.replace(r'_', '')
    text = text.replace(r')', '')
    text = text.replace(r'(', '')
    text = text.replace(r':', '')
    #text = text.replace(r'?', '')
    text = text.replace(r'%', '')
    text = text.replace(r'+', '')
    text = text.replace(r'=', '')
    text = text.replace(r'&', '')
    text = text.replace(r'^', '')
    text = text.replace(r'$', '')
    text = text.replace(r'#', '')
    #text = text.replace(r'!', '')
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
    
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove emails
    text = re.sub(r'\s+', ' ', text)  # Remove newline chars
    text = re.sub(r"\'", "", text)  # Remove single quotes
    text = re.sub(r"http[s]?\://\S+","",text) #Remove http:
    text = re.sub(r"[0-9]", "",text) # Remove number
    text = re.sub(r'\s+',' ',text) # Remove space
    text = text.encode('ascii', 'ignore').decode()
    
    return text

# Extrac text from Scanned PDF
def extractTextFromScannedPDFs(fname):
    text = ""
    while True:
        pages = convert_from_path(fname)
        try:
            for page in pages:
                text += pytesseract.image_to_string(page)
            new_text = text.strip("\n")
        except:
            print("Oops!  PDF not readable")
            break
    return new_text

# Extrac text from Generated PDF
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

# Extract text for given pdf
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

# Images Extraction
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# Save the images extracted from a PDF into one PDF file
def saveImagesIntoPdf(file_path, images_path, error):
    try:
        print(f'images_path:{images_path}')
        doc = fitz.open()  # PDF with the pictures
        imgdir = "img"  # Where the images are
        imglist = os.listdir(imgdir)  # List of images
        imgcount = len(imglist)  # Images count

        for i, f in enumerate(imglist):
            img = fitz.open(os.path.join(imgdir, f))  # Open image as document
            rect = img[0].rect  # Image dimension
            pdfbytes = img.convert_to_pdf()  # Make a PDF stream
            img.close()  # Close the image
            imgPDF = fitz.open("pdf", pdfbytes)  # Open stream as PDF
            page = doc.new_page(width = rect.width,height = rect.height)  # Image dimension
            page.show_pdf_page(rect, imgPDF, 0)  # Image fills the page
        doc.save(images_path)
    except:
        error = True
        print("Error occurred when Saving imagges into one pdf file")
        
    return error

# Extract images from files
def getImagesFromPDF(file_path, images_path):
    # Make img directory to save images extrated from PDF temporary 
    imgdir = "img"
    error = False
    # Check if imgdir already exists
    if os.path.exists(imgdir):
        shutil.rmtree(imgdir)
        path = os.path.join("", imgdir)
        os.mkdir(path)
        print("Directory img created")
    # Create imgdir if not exist
    else:
        path = os.path.join("", imgdir)
        os.mkdir(path)
        print("Directory img created")

    # Extract Images from currgivenent PDF and save them under directory img
    try:    
        doc = fitz.open(file_path)
        imagelist = []
        print(type(doc))
        for i in range(len(doc)):
            for img in doc.get_page_images(i):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:       
                    pix.save(f"{imgdir}/p%s-%s.png" % (i, xref))
                else:              
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    imagelist.append(pix1)
                    pix1.save(f"{imgdir}/p%s-%s.png" % (i, xref))
                    pix1 = None
                pix = None
    except:
        error = True
        print("Error occurred when extracting imagges")

    # Save all the images extracted into one pdf file
    # Check for error
    error = saveImagesIntoPdf(file_path, images_path, error)
    return error

# Extract images for all the papers
def extractImagesFromFile(paperList_textList, paperDirName, imageDirName):
    paperList = paperList_textList.paper_path
    images_path_list = []
    # Extract Images from the paper
    # ---------------------------------------------------------------------------
    for paper in paperList:
        images_path =  paper.replace(paperDirName, imageDirName)
        print(f'images_path: {images_path}')
        print(f"images_path: {images_path}")
        if (getImagesFromPDF(paper, images_path) == False):
            print("Successuflly extracted images")
        else:
            print("Failed to extract image")
        images_path_list.append(images_path)
    # ---------------------------------------------------------------------------
    return images_path_list

# Check if file is a paper and extract text, then save them
def extractPaperInfo(files, pdfDirName, paperDirName, txtDirName):

    paper_path_list = []
    text_file_list = []
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
                print(f'abstract_pos: {abstract_pos},  references_pos: {references_pos}')
                # if the article contains abstract and reference, count it as a paper
                if (references_pos != -1 or abstract_pos != -1 ):

                    # Make a copy of this paper and save it under folder {paperDirName}
                    # ---------------------------------------------------------------------------
                    shutil.copy(file, paperDirName)
                    paper_path =  file.replace(pdfDirName, paperDirName)
                    paper_path_list.append(paper_path)
                    # ---------------------------------------------------------------------------

                    # Save the text as text document
                    # ---------------------------------------------------------------------------
                    print(f'file : {file}')
                    text_file_name = file.replace(pdfDirName, txtDirName)
                    text_file_name = text_file_name.replace('.pdf', '.txt')
                    print(f'text_file_name : {text_file_name}')
                    # text_file_name = text_file_name[1].split('.')
                    # text_file_name = f"{txtDirName}/{text_file_name[0]}.txt"
                    with open(text_file_name, 'w') as f:
                        f.write(text)
                        f.close() 
                    text_file_list.append(text_file_name)
                    # ---------------------------------------------------------------------------

        except:
                print('Error occurred')

    return paper_path_list, text_file_list  #, images_path_list
###############################################################################################################
###############################################################################################################

def getLinkToPath(paper_path_list, text_path_list, image_list):

    base_url = "http://127.0.0.1:5000/"
    new_paper_path_list = []
    new_text_path_list = []
    new_image_list = []
    for paper in paper_path_list:
        new_paper_path_list.append(f'{base_url}{paper}')

    for text in text_path_list:
        new_text_path_list.append(f'{base_url}{text}')
    
    for iamge in image_list:
        new_image_list.append(f'{base_url}{iamge}')

    new_paperList_textList_imageList =  {'paper_path': new_paper_path_list, 'text_path': new_text_path_list, 'image_path': new_image_list}
    new_paperList_textList_imageLists = pd.DataFrame(new_paperList_textList_imageList)
    
    return new_paperList_textList_imageLists

# Main method
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def extractInfoFromPapers(userInputTopic, lang="en"):

    pdfDirName, paperDirName, imageDirName, csvDirName, txtDirName, ldaDirName = makeNewDiretoryForGivenTopic(userInputTopic)
    topic = userInputTopic.replace(" ", "")
    ################################# Data Processing Section ################################################
    # User can add articles into the folder name {pdfDirName}
    # ----------------------------------------------------------------------------------------------------
    # Get All the files under directory {pdfDirName}
    files_path  = [pdfDirName + '/' +name for name in os.listdir(pdfDirName) if name.endswith('pdf')]
    # 
    # Extract research pappers from the files and save them under folder name {paperDirName}
    # Convert them into text document and save them under folder name {txtDirName}
    paper_path_list, text_path_list = extractPaperInfo(files_path, pdfDirName, paperDirName, txtDirName)
    # 
    # Create a dictionary
    paperList_textList = {'paper_path': paper_path_list, 'text_path': text_path_list}
    # Build pandas dataframe for dictionary
    paperList_textList = pd.DataFrame(paperList_textList)
    #  Save the data into csv file
    paperList_textList.to_csv(f'{csvDirName}/paperList_textList.csv', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    # 
    # Extract Images for every papers and save them under folder name {imageDirName}
    image_list = extractImagesFromFile(paperList_textList, paperDirName, imageDirName)
    paperList_textList_imageList = {'paper_path': paper_path_list, 'text_path': text_path_list, 'image_path': image_list}
    paperList_textList_imageLists = pd.DataFrame(paperList_textList_imageList)
    paperList_textList_imageLists_csv = f'{csvDirName}/paperList_textList_imageList.csv'
    paperList_textList_imageLists.to_csv(paperList_textList_imageLists_csv, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')


    new_paperList_textList_imageList = getLinkToPath(paper_path_list, text_path_list, image_list)
    new_paperList_textList_imageList_html = f'{csvDirName}/paperList_textList_imageList.html'
    HTML(new_paperList_textList_imageList.to_html(new_paperList_textList_imageList_html, render_links=True, escape=False)) 
    # # #-----------------------------------------------------------------------------------------------------
    ###########################################################################################################

    total_num_paper = len(files_path)
    num_paper_left = len(paper_path_list)
    return_list  = []
    return_list.append(f'Number of files processed: {total_num_paper}')
    return_list.append(f'Number of paper found: {num_paper_left}')
    table_name_list = []
    table_name_list.append(new_paperList_textList_imageList_html)
    return return_list, table_name_list
    

###############################################################################################################
###############################################################################################################