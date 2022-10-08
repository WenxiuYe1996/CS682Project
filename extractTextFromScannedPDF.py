from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import numpy as np

#code to extract text from scanned pdfs
pages = convert_from_path('a.pdf')

i = 0
for page in pages:
    page.save(f'out{i}.png', 'PNG')
    i+=1
    img1 = np.array(Image.open(f'out{i}.png'))
    text = pytesseract.image_to_string(img1)
    print(text)
