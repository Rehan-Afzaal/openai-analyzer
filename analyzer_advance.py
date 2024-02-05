import logging
import os
import tempfile
import traceback
from threading import Thread
import openai
import pytesseract
import requests
from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from pdf2image import convert_from_path
from pymongo import MongoClient
from constants import OPERATING_AGREEMENT_PROMPT, PRIVATE_PLACEMENT_PROMPT, OWNER_DISTRIBUTION_PROMPT, WATERFALL_PROMPT, \
    GENERAL_PROMPT_ANALYSIS, MONEY_MARKET_PROMPT, SYSTEM_PROMPT
import metadata

app = Flask(__name__, static_folder='static')
CORS(app)
app.debug = True

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    extracted_text = '\n'.join([pytesseract.image_to_string(img) for img in images])
    return extracted_text


# def extract_text_from_pdf(file_stream):
#     images = convert_from_path(file_stream)
#     extracted_text = '\n'.join([pytesseract.image_to_string(img) for img in images])
#     return extracted_text


# Path to your PDF
# source env/bin/activate
@app.route('/analyzer', methods=['GET', 'POST'])
def analyzerScript():
    logging.info("Analyzer script accessed")
    app.config['TIMEOUT'] = 500
    # userId="123456789"
    propertyId = request.form.get('propertyId')
    docType = request.form.get('docType')
    promptType = request.form.get('promptType', 'private placement')
    # promptType = request.form.get('promptType', 'PPM')  # Default to 'PPM' if not provided requestedprompt = "You
    # are Contract Analysis Expert in a Fund Administrator position, your role involves analyzing and key information
    # from operating agreements and other contracts for a Fund Administrator role. Your expertise is crucial in
    # interpreting legal and financial documents, identifying key clauses, and presenting findings in a structured
    # JSON format."
    if request.method == 'GET':
        return "Server Running"
    if 'file' not in request.files:
        return "Please attach file"
    if 'userId' in request.form:
        userId = request.form.get('userId')
    file = request.files['file']
    if file.filename == '':
        return "Please attach file"
    # file.save(file.filename)
    # print(request.files.getlist('file')[0])
    for file in request.files.getlist('file'):
        # print("filename",file.filename)
        file.save(file.filename)
        send_data_to_server(userId, file.filename, {}, "Pending", propertyId, docType)

    for file in request.files.getlist('file'):
        # print('File successfully uploaded!', file.filename)
        send_data_to_server(userId, file.filename, {}, "Processing", propertyId, docType)
        pdf_path = './' + file.filename

        # Extract text
        content = extract_text_from_pdf(pdf_path)
        # print(content)

        #  Save the extracted text to a file
        # ocr_output_filename = f"{file.filename}_ocr.txt"
        # with open(ocr_output_filename, "w") as text_file:
        # text_file.write(content)
        # print(f"OCR output saved to {ocr_output_filename}")
        
        # prompts = []
        # logging.info(f"promptType from FE, {promptType}")

        # if promptType == 'Operating Agreement':
        #     prompts.append(f'{OPERATING_AGREEMENT_PROMPT}\n{content}')
        # elif promptType == 'private placement':
        #     prompts.append(f'{PRIVATE_PLACEMENT_PROMPT}\n{content}')

        # elif promptType == 'owner distributions':
        #     prompts.append(f'{OWNER_DISTRIBUTION_PROMPT}\n{content}')
        # if promptType == 'Operating Agreement':
        #     prompts(f'{WATERFALL_PROMPT}\n{content}')
        # else:
        #     return "Invalid Prompt"
        prompts = []
        logging.info(f"promptType from FE, {promptType}")

        if promptType == 'Operating Agreement':
            prompts.append(f'{OPERATING_AGREEMENT_PROMPT}\n{content}')
            prompts.append(f'{WATERFALL_PROMPT}\n{content}')
        elif promptType == 'private placement':
            prompts.append(f'{PRIVATE_PLACEMENT_PROMPT}\n{content}')
        elif promptType == 'owner distributions':
            prompts.append(f'{OWNER_DISTRIBUTION_PROMPT}\n{content}')
        else:
            return "Invalid Prompt"

        # Load environment variables from .env file
        load_dotenv()
        # Use the API key
        # api_key = os.getenv('OPENAI_API_KEY')

        # Initialize the OpenAI client
        client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        # Send the prompt to the GPT model
        print("prompt size", len(prompts))
        for prompt in prompts:
            logging.info(f"prompt from for loop: {prompt}")
            response = client.chat.completions.create(
                model='gpt-4-1106-preview',
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
                stop=None,
                seed=123789,
                response_format={"type": "json_object"},
                messages=[{'role': 'user', 'content': prompt}]
            )

            logging.info(f"response from gpt: {response}")

            # Extract and print the model's response
            processed_content = response.choices[0].message.content.replace('json```', '');
            processed_content = processed_content.replace('json ', '')
            processed_content = processed_content.replace('```', '')

            send_data_to_server(userId, file.filename, processed_content, "Processed", propertyId, docType)
    return "Files are being processed"

def send_data_to_server(userId, filename, processedData, currentStatus, propertyId, docType):
    # Get data from the request or any other source
    if currentStatus == "Processed":
        file_path = './' + filename
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"File not found: {file_path}")

    try:
        # print("processedData",processedData)
        # print("json",userId,filename,docType)
        node_backend_url = os.getenv('NODE_BACKEND_URL')

        response = requests.post(node_backend_url,
                                 json={"userId": userId, "filename": filename, "processedData": processedData,
                                       "currentStatus": currentStatus, "propertyId": propertyId,
                                       "docType": docType})  # You can use 'data' as a dictionary or other data types

        if response.status_code == 200:
            # Request was successful
            logging.info("Data sent successfully to server: %s",  response.content)
            # print("Success")
            return {}  # If the server returns JSON response
        else:
            # Handle errors
            logging.info(f"Request failed with status code {response.status_code}")
            # print(response.text)  # Print the response content for debugging
            return None  # You can return an error indicator

    except requests.exceptions.RequestException as e:
        # Handle exceptions
        logging.info(f"An error occurred: {str(e)}")
        return None  # You can return an error indicator
    # print("post form data123:", requests.post)


# Path to your PDF
# source env/bin/activate
@app.route('/moneymarket', methods=['GET', 'POST'])
def marketScript():
    app.config['TIMEOUT'] = 500  # Set a custom timeout for requests
    # userId="123456789"
    propertyId = request.form.get('propertyId')  # Get property ID from form data
    promptType = request.form.get('promptType', 'Money Market')  # Get prompt type, default to 'Money Market'
    # moneyMarketId = request.form.get('moneyMarketId') # Get money market ID from form data

    if request.method == 'GET':
        return "Server Running"

    if 'file' not in request.files:
        return "Please attach file"

    if 'userId' in request.form:
        userId = request.form.get('userId')

    all_text = ""

    for file in request.files.getlist('file'):
        # print("filename", file.filename)
        file.save(file.filename)
        send_data_to_server_market(userId, file.filename, {}, "Pending",
                                   propertyId)  # Send data to server with status "Pending"

        # Process each file for OCR
        pdf_path = './' + file.filename  # Path to the saved file
        content = extract_text_from_pdf(pdf_path)  # Extract text from the file
        all_text += content + "\n"  # Append the content to the all_text variable
        # print(all_text)

        # Extract text
        # content = extract_text_from_pdf(pdf_path)
        # print(content)

        #  #Save the extracted text to a file
        # ocr_output_filename = f"{file.filename}_ocr.txt"
        # with open(ocr_output_filename, "w") as text_file:
        #     text_file.write(content)
        # print(f"OCR output saved to {ocr_output_filename}")

        if promptType == 'Money Market':
            prompt = f'''
            {MONEY_MARKET_PROMPT}
            {all_text}
            '''
        # Load environment variables from .env file
        # load_dotenv()
        # Use the API key
        # api_key = os.getenv('OPENAI_API_KEY')

        # Initialize the OpenAI client
        client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        try:
            response = client.chat.completions.create(
                model='gpt-4-1106-preview',
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
                stop=None,
                seed=123789,
                response_format={"type": "json_object"},
                messages=[{'role': 'user', 'content': prompt}]
            )
            processed_content = response.choices[0].message.content.replace('json```', '')
        except Exception as e:
            return f"Error in processing GPT-4 response: {str(e)}"
        for file in request.files.getlist('file'):
            send_data_to_server_market(userId, file.filename, processed_content, "Processed", propertyId)
        else:
            return "Invalid Prompt"
        return "Files have been processed"


def send_data_to_server_market(userId, filename, processedData, currentStatus, propertyId):
    file_path = './' + filename

    # Check if the file exists before attempting to delete
    if currentStatus == "Processed" and os.path.exists(file_path):
        os.remove(file_path)  # Delete the file

    try:
        # print("processedData", processedData)
        json_data = {"userId": userId, "filename": filename, "processedData": processedData,
                     "currentStatus": currentStatus, "propertyId": propertyId}
        node_backend_url = os.getenv('NODE_BACKEND_URL')
        response = requests.post(node_backend_url, json=json_data)
        if response.status_code == 200:
            logging.info("Data sent successfully to server.")
            return {}  # If the server returns JSON response
        else:
            logging.error("Failed to send data to server. Status code: %d, Response: %s", response.status_code, response.text)

            # print(response.text)
            return None

    except requests.exceptions.RequestException as e:
        # print(f"An error occurred: {str(e)}")
        return None


# Path to your PDF
# source env/bin/activate
@app.route('/promptsanalysis', methods=['GET', 'POST'])
def promptsAnalyzer():
    app.config['TIMEOUT'] = 500
    # userId="123456789"
    # propertyId=request.form.get('propertyId')
    promptType = request.form.get('promptType',
                                  default="You are an investment analyst with extensive experience in private placements. Your role is to aid in constructing a analysis text in JSON format for private placement offerings, making complex financial information accessible and comprehensible to potential investors, follow these steps to extract and organize the data into JSON format")

    if request.method == 'GET':
        return "Server Running"
    if 'file' not in request.files:
        return "Please attach file"
    if 'userId' in request.form:
        userId = request.form.get('userId')

    documentId = request.form.get('documentId')  # added by asad to send to server

    file = request.files['file']
    if file.filename == '':
        return "Please attach file"

    for file in request.files.getlist('file'):
        # print("filename",file.filename)
        file.save(file.filename)
        send_data_to_server_prompts(userId, file.filename, documentId, {}, "Pending")  # documentId
    for file in request.files.getlist('file'):
        # print('File successfully uploaded!', file.filename)
        send_data_to_server_prompts(userId, file.filename, documentId, {}, "Processing")
        pdf_path = './' + file.filename  # in
        # Extract text
        content = extract_text_from_pdf(pdf_path)

        prompt = f'''       
        {promptType}
        {GENERAL_PROMPT_ANALYSIS}
        {content}
        '''
        # Load environment variables from .env file
        load_dotenv()
        # Use the API key
        api_key = os.getenv('OPENAI_API_KEY')
        # Initialize the OpenAI client
        client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        # Send the prompt to the GPT model
        response = client.chat.completions.create(
            model='gpt-4-1106-preview',
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            stop=None,
            seed=123789,
            response_format={"type": "json_object"},
            messages=[{'role': 'user', 'content': prompt}]
        )
        # Extract and print the model's response
        processed_content = response.choices[0].message.content.replace('json```', '');
        processed_content = processed_content.replace('json ', '')
        processed_content = processed_content.replace('```', '')
        send_data_to_server_prompts(userId, file.filename, documentId, processed_content, "Processed")
    return "Files are being processed"


def send_data_to_server_prompts(userId, filename, documentId, processedData, currentStatus):
    # Get data from the request or any other source
    if currentStatus == "Processed":
        os.remove('./' + filename)
    try:
        # print("processedData",processedData)
        node_backend_url = os.getenv('NODE_BACKEND_URL')
        response = requests.post(node_backend_url,
                                 json={"userId": userId, "documentId": documentId, "filename": filename,
                                       "processedData": processedData,
                                       "currentStatus": currentStatus})  # You can use 'data' as a dictionary or other data types
        if response.status_code == 200:
            # Request was successful
            # print("Success")
            return {}  # If the server returns JSON response
        else:
            # Handle errors
            # print(f"Request failed with status code {response.status_code}")
            # print(response.text)  # Print the response content for debugging
            return None  # return an error indicator
    except requests.exceptions.RequestException as e:
        # Handle exceptions
        # print(f"An error occurred: {str(e)}")
        return None  # You can return


#  Handles text extraction and chunking in a separate thread to avoid blocking the main application.
def background_processing(file_path, user_id, document_id):
    try:
        logging.info(f"Starting background processing for file: {file_path}")
        # Extract text from PDF File
        text = extract_text_from_pdf_chat(file_path)
        if text:
            # process the text into chunks and store into mongoDB
            chunks = process_text_to_chunks(text)
            logging.info(f"Chunks generated: {len(chunks)}")
            logging.info(f"Updating MongoDB with chunks for document ID: {document_id}")
            update_text_chunks(document_id, chunks, os.path.basename(file_path), user_id)
        else:
            logging.error(f"No text extracted from file: {file_path}")
        # Remove Temporary files
        os.remove(file_path)
        logging.info(f"Temporary file removed: {file_path}")
    except Exception as e:
        logging.error(f"Error in background processing: {e}")
        logging.error(traceback.format_exc())


# Load environment variables
load_dotenv()
import certifi

ca = certifi.where()

# MongoDB setup
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri, tlsCAFile=ca)
db_name = os.getenv('DB_NAME')

file_collection = client.get_database(db_name).get_collection("chats")

# OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')


# @app.route('/test', methods=['GET'])
# def test():
#     print(file_collection.find_one())
#     return ""


@app.route('/upload', methods=['POST'])
def upload_file():
    logging.info("Upload route accessed")
    # check if 'file' is in the request
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    # process the uploaded file
    uploaded_file = request.files['file']
    user_id = request.form.get('user_id', '')
    # check file type and save into temporary location
    file_type = uploaded_file.filename.split('.')[-1].lower()
    if file_type != 'pdf':
        logging.error("Unsupported file type")
        return jsonify({"error": "Unsupported file type"}), 400

    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    uploaded_file.save(temp_file)
    temp_file.close()

    # Create initial document entry and start background processing thread
    document_id = create_initial_document_entry(uploaded_file.filename, user_id)
    logging.info(f"Initial document entry created with ID: {document_id}")

    thread = Thread(target=background_processing, args=(temp_file.name, user_id, document_id))
    thread.start()

    logging.info("Started background processing thread")
    return jsonify({"message": "File upload received, processing started", "document_id": str(document_id)})


# Create initial document entry in MongoDB
def create_initial_document_entry(filename, user_id):
    initial_document = {"filename": filename, "content": [], "user_id": ObjectId(user_id)}
    result = file_collection.insert_one(initial_document)
    return result.inserted_id


# Update text chunks in MongoDB
def update_text_chunks(document_id, chunks, filename, user_id):
    try:
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise ValueError("Chunks must be a list of strings")
        logging.info(f"Performing MongoDB update for document ID: {document_id}")
        file_collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"content": chunks, "filename": filename, "user_id": ObjectId(user_id)}}
        )
        logging.info(f"MongoDB update complete for document ID: {document_id}")
    except Exception as e:
        logging.error(f"Error in updating MongoDB: {e}")
        logging.error(traceback.format_exc())


# Route for handling user quries based on document content
@app.route('/chat', methods=['POST'])
def chat():
    # check if required Json data is available in the request
    if not request.json or 'question' not in request.json or 'document_id' not in request.json or 'user_id' not in request.json:
        return jsonify({'error': 'Missing data'}), 400

    # Extract relevant data from the request
    user_question = request.json['question']
    document_id = request.json['document_id']
    user_id = request.json['user_id']
    prompt = f"{system_prompt}{user_question}"
    # Retrieve document from MongoDB
    bson_document_id = ObjectId(document_id)
    document = file_collection.find_one({"_id": bson_document_id})
    if not document:
        return jsonify({"error": "Document not found"}), 404

    # Initiate chatbot and update MongoDB with user query and chatbot response
    # response = init_chatbot(document['content'], user_id)({'question': user_question},)(prompt)
    response = init_chatbot(document['content'], user_id, prompt)  # Pass prompt as argument

    file_collection.update_one({"_id": bson_document_id}, {"$push": {
        "chat_messages": {"user_id": user_id, "question": user_question, "response": response.get("answer")}}})
    return jsonify({"response": response.get("answer")})


system_prompt = {SYSTEM_PROMPT}


# Route for checking server status
@app.route("/", methods=["GET"])
def index():
    return "Server is running.."


# Extract text from a PDF file using Tesseract OCR
def extract_text_from_pdf_chat(file_path):
    try:
        # Implementation using pytesseract and pdf2image
        logging.info(f"Extracting text from PDF: {file_path}")
        images = convert_from_path(file_path)
        text = ''.join(pytesseract.image_to_string(image) for image in images)
        logging.info(f"Extracted text length: {len(text)} characters")
        return text
    except Exception as e:
        logging.error(f"Error in extracting text from PDF: {e}")
        logging.error(traceback.format_exc())
        return ""


# Process text into chunks using a custom text splitter
# def process_text_to_chunks(text):
#     text_splitter = CharacterTextSplitter(separator=' ', chunk_size=500, chunk_overlap=150,strip_whitespace=False, length_function=len)
#     return text_splitter.split_text(text)
# Example of a modified process_text_to_chunks function
def process_text_to_chunks(text):
    # Use NLP library for semantic splitting (e.g., spaCy)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = [str(span) for span in doc.sents]  # Splitting by sentences as an example
    return chunks

# Store text chunks in MongoDB
def store_text_chunks(chunks, filename, user_id):
    document = {"filename": filename, "content": chunks, "user_id": ObjectId(user_id)}
    result = file_collection.insert_one(document)
    return result.inserted_id


def init_chatbot(text_chunks, user_id, prompt):
    # Retrieve chat history from MongoDB, handling cases where no history exists
    chat_history = file_collection.find_one({"_id": ObjectId(user_id)})
    past_messages = chat_history["chat_messages"] if chat_history else []

    # Create text embeddings for efficient knowledge base search
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Build a knowledge base using FAISS for fast retrieval
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    # Initialize the GPT model for generating responses
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4-1106-preview", temperature=0.1)

    # Establish conversational memory to store past messages
    memory = ConversationBufferMemory(memory_key="chat_history", initial_messages=past_messages)
    # Construct the retrieval chain, combining knowledge base, GPT model, and memory
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=knowledge_base.as_retriever(), memory=memory)
    response = chain(prompt)
    return response
# from sentence_transformers import SentenceTransformer
# import faiss
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.docstore.in_memory import InMemoryDocstore

# def init_chatbot(text_chunks, user_id, prompt):
#     # Retrieve chat history from MongoDB
#     chat_history_doc = file_collection.find_one({"_id": ObjectId(user_id)})
#     past_messages = chat_history_doc["chat_messages"] if chat_history_doc else []
    
#     # Create text embeddings
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode(text_chunks)
    
#     # Build a knowledge base using FAISS
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
    
#     docstore = InMemoryDocstore()
#     index_to_docstore_id = {i: str(i) for i in range(len(text_chunks))}  # Example mapping
#     embedding_function = lambda texts: model.encode(texts)
    
#     knowledge_base = FAISS(embedding_function, index, docstore, index_to_docstore_id)
    
#     # Initialize the GPT model
#     llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4-1106-preview", temperature=0.1)
    
#     # Establish conversational memory
#     memory = ConversationBufferMemory(memory_key="chat_history", initial_messages=past_messages)
    
#     # Construct the retrieval chain
#     chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=knowledge_base.as_retriever(), memory=memory)
    
#     # Generate response
#     response = chain(prompt)
#     return response

@app.route('/process_pdf', methods=['POST'])
def process_pdf_post():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Please attach file"
        else:
            uploaded_file = request.files['file']
            temp_dir = 'temp_upload'
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.filename)
            uploaded_file.save(file_path)
            json_return = metadata.process_pdf(file_path)
            os.remove(file_path)
            return json_return
    else:
        logging.info("Not a Post Request")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1338)
