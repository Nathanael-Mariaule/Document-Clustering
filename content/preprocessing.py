import spacy

def preprocessing(text, nlp, gpu_enable=False) -> str:
    """
        remove '\\n' caracters from a text, use lemmatization and stop_words from spacy and finally return a list of the remaining words
        :param str text: text to be transformed
        :param spacy.lang.en.English nlp: 'en_core_web_sm' module from spacy
        :param bool gpu_enable: If True, spacy will run using GPU
        :return str: cleaned text
    
    """
    if gpu_enable:
        spacy.require_gpu()
    text = str(text)
    text = text.replace('\\n', ' ')
    tokens = nlp(text)
    tokens = [token.lemma_ for token in tokens if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.lemma_.strip()!= '')]
    cleaned = ' '.join([word.lower() for word in tokens if word.isalpha() and len(word)>2])
    return cleaned
