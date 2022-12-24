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
import requests

def getImagesFromPDF(file_path):

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


# Main method
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def getImagesFromFile(userInputArticleLink):

     
    response = requests.get(userInputArticleLink, timeout=20)
    print('data recived')
    
    path = 'file/a.pdf'
    open(path, "wb").write(response.content)
    
    if (getImagesFromPDF(path)):
        print("Successfully extracted images")
    else:
        print("Error occured")

    return_list = []

    return return_list
    

###############################################################################################################
###############################################################################################################