import spacy

def preprocessing(text, nlp, gpu_enable=False):
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
