from flask import request, jsonify
import fitz  # PyMuPDF
import spacy
from transformers import pipeline
import logging
import os

# Initialize Spacy and Summarizer
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="t5-small")

# Set up logging
logging.basicConfig(level=logging.INFO)


def process_pdf(file):
    try:
        # Process PDF file
        text = extract_pdf(file)
        entities = nlp_analysis(text)
        summary = summarize_text(text)

        return jsonify({
            'metadata': extract_pdf_metadata(file),
            'entities': entities,
            'summary': summary
        })
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return jsonify({'error': str(e)}), 500


def extract_pdf_metadata(file):
    """
    Extract metadata from a PDF file.

    :param file: Either a file stream or a file path
    :return: Metadata dictionary
    """
    logging.info("Extracting PDF metadata")
    try:
        with open(file, 'rb') as file_stream:
            doc = fitz.open(stream=file_stream.read(), filetype="pdf")
            metadata = doc.metadata
            doc.close()
        return metadata
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
        raise


def extract_pdf(file):
    """
    Extract text from a PDF file.

    :param file: Either a file stream or a file path
    :return: Extracted text as a string
    """
    logging.info("Extracting text from PDF")
    try:
        with open(file, 'rb') as file_stream:
            doc = fitz.open(stream=file_stream.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        return text
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        raise


def nlp_analysis(text):
    """
    Perform NLP analysis to extract entities from text.

    :param text: Text to analyze
    :return: List of entities
    """
    logging.info("Performing NLP analysis")
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except Exception as e:
        logging.error(f"Error in NLP analysis: {e}")
        raise


def summarize_text(text):
    """
    Summarize the given text.

    :param text: Text to summarize
    :return: Summary of the text
    """
    logging.info("Summarizing text")
    try:
        summary = summarizer(text, max_length=130, truncation=True, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logging.error(f"Error in text summarization: {e}")
        raise