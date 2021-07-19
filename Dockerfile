FROM python:3.7.11-slim-buster

# Install the security updates.
RUN apt-get update && apt-get -y upgrade

# Install the required dependencies.
RUN apt-get -y install tesseract-ocr && apt-get -y install poppler-utils

# Remove all cached file. Get a smaller image.
RUN apt-get clean && \ 
    rm -rf /var/lib/apt/lists/*
    
# Copy the application.
COPY . /app
WORKDIR /app



# Install the app librairies.
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm


# Open Port Start the app.
EXPOSE 8501
CMD exec streamlit run content/app.py
