import pickle
from PIL import Image
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
import os
import spacy
from preprocessing import preprocessing
import pandas as pd
import csv


def predict(text, clusters_label, vectorizers, kmeans) -> int:
    """
        Predict the label of the input text. The label is computed using a sequence of k-mean algorithm. The label is the first predicted label that fall into the list clusters_label
        :param str text: contains the text to be labelled
        :param List[List[int]] clusters_label:  clusters_label[i] is the least of labels that are accepted by kmeans[i]. Such label as value>= len(clusters_label[i-1])
        :param List[sklearn.feature_extraction.text.TfidfVectorizer] vectorizers: list of TfidfVectorizer to convert text into a vector to use k-mean algorithms
        :param List[sklearn.cluster.KMeans] kmeans: list of k-mean algorithm use to predict the labels
        :return int: Value of the label predicted
    """
    nlp = spacy.load("en_core_web_sm")
    text = preprocessing(text, nlp) #lemmatization and remove stopwords
    for i in range(len(clusters_label)):
        vector = vectorizers[i].transform([text])
        label = kmeans[i].predict(vector)
        if i==0 and label[0] in clusters_label[i]:  #the label is in the accepted value
            return clusters_label[i].index(label[0])
        elif i>0 and label[0]+len(clusters_label[i-1]) in clusters_label[i]:  #the label is in the accepted value for kmeans[i]
            return clusters_label[i].index(label[0]+len(clusters_label[i-1]))
    return len(clusters_label[-1]) #default label if no other has been found



def pdf_to_text(pdfpath)->str:
    """
        Convert the text content of a pdf into a string
        :param str or bytes pdfpath: path to pdf file as string or pdf into bytes format
        :return str: pdf into str format
    """
    # This function was inspired from
    #https://www.geeksforgeeks.org/python-reading-contents-of-pdf-using-ocr-optical-character-recognition/
    if type(pdfpath)== str:
        pages = convert_from_path(pdfpath, 500)
    else:
        pages = convert_from_bytes(pdfpath, 500)

    image_counter = 1
    #save each page of the pdf into a jpg file
    for page in pages:
	    filename = "page_"+str(image_counter)+".jpg"
	    page.save(filename, 'JPEG')
	    image_counter = image_counter + 1
    filelimit = image_counter-1
    #add the text content of each image to a string
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

    