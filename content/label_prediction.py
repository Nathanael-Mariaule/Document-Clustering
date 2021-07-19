import pickle
from PIL import Image
import pytesseract
import sys
from pdf2image import convert_from_path, convert_from_bytes
import os
import spacy
from preprocessing import preprocessing
import pandas as pd
import csv


def predict(text, clusters_label, vectorizers, kmeans):
    nlp = spacy.load("en_core_web_sm")
    text = preprocessing(text, nlp)
    for i in range(len(clusters_label)):
        vector = vectorizers[i].transform([text])
        label = kmeans[i].predict(vector)
        if i==0 and label[0] in clusters_label[i]:
            return clusters_label[i].index(label[0])
        elif i>0 and label[0]+len(clusters_label[i-1]) in clusters_label[i]:
            return clusters_label[i].index(label[0]+len(clusters_label[i-1]))
    return len(clusters_label[-1])



def pdf_to_text(pdfpath):
    ##https://www.geeksforgeeks.org/python-reading-contents-of-pdf-using-ocr-optical-character-recognition/
    if type(pdfpath)== str:
        pages = convert_from_path(pdfpath, 500)
    else:
        pages = convert_from_bytes(pdfpath, 500)

    image_counter = 1

    for page in pages:
	    filename = "page_"+str(image_counter)+".jpg"
	    page.save(filename, 'JPEG')
	    image_counter = image_counter + 1
    filelimit = image_counter-1
    text = ""
    for i in range(1, filelimit + 1):
        filename = "page_"+str(i)+".jpg"
        text += str(((pytesseract.image_to_string(Image.open(filename)))))
        os.remove(filename)
    text = text.replace('-\n', '')
    return text



if __name__=='__main__':
    with open('content/model.pickle', 'rb') as file:
        model = pickle.load(file)

    clusters_label = model[0] 
    clusters_name = model[1]
    cluster_relevant_words = model[2]
    vectorizers = model[3] 
    kmeans = model[4]
    pdfpath = 'content/Documents/1ddfe560c9b26c5992e838e834525f88.pdf'
    #text = pdf_to_text(pdfpath)
    df = pd.read_csv('content/Documents/test', quoting=csv.QUOTE_NONE, sep='\t', header=None)
    nlp = spacy.load("en_core_web_sm")
    print(df.iloc[0,0])
    text = preprocessing(df.iloc[0,3], nlp)
    print(predict(text, clusters_label, vectorizers, kmeans))

    