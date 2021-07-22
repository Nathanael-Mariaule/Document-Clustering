import streamlit as st
import pandas as pd
from label_prediction import predict, pdf_to_text
import spacy
import pickle



st.title('Clustering of the Kleister Charity Dataset')

'''
    The Kleister Charity Dataset is dataset that contains PDF files published by British charities.
    The dataset is available on [GitHub](https://github.com/applicaai/kleister-charity) and was extracted from https://www.gov.uk/government/organisations/charity-commission.

    As project during AI BeCode Training, I was asked to use clustering algorithms to give structure to this dataset. Using k-mean, I splitted the dataset into the following categories:
'''

clusters_name = {
                    0 : 'School/College',
                    1 : 'Church',
                    2 : 'Club',
                    3 : 'Art/Festival',
                    4 : 'Camp/scout',
                    5 : 'University/research',
                    6 : 'International',
                    7 : 'Pension',
                    8 : 'River',
                    9 : 'Park/Museum',
                    10 : 'School/Optional',
                    11 : 'Examination',
                    12 : 'Liabililities/Policies',
                    13 : 'Other'
                }

clusters_name = pd.Series(clusters_name)
st.table(clusters_name)




with open('content/model.pickle', 'rb') as file:
        model = pickle.load(file)

clusters_label = model[0] 
clusters_name = model[1]
cluster_relevant_words = model[2]
vectorizers = model[3] 
kmeans = model[4]
nlp = spacy.load("en_core_web_sm")

uploaded_file = st.file_uploader("Choose a odf file from de Kleister Charity Dataset")
left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Category of the file:')
if pressed:
    if not uploaded_file:
        right_column.write('Please, upload a file')
    else:
        text = pdf_to_text(uploaded_file.getvalue())
        label = predict(text, clusters_label, vectorizers, kmeans)
        label_category = clusters_name[label]
        right_column.write(f'{label} : {label_category}')

'''
    More details and the python codes are available [on my GitHub](https://github.com/Nathanael-Mariaule/Document-Clustering)
'''
