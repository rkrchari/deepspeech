import os
import asyncio
import faiss
from flask import Flask, render_template, request, redirect, session
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session

# Set up the upload folder and allowed file types
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Map for different vector stores based on PDF format
VECTORSTORE_PATHS = {
    'format_a': 'faiss_index_format_a.index',
    'format_b': 'faiss_index_format_b.index'
}

vectorstore = None
conversation_chain = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n', ' ', ''],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks, format_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    save_vectorstore(vectorstore, format_key)  # Save with format key
    return vectorstore

def save_vectorstore(vectorstore, format_key):
    faiss.write_index(vectorstore.index, VECTORSTORE_PATHS[format_key])

def load_vectorstore(format_key):
    index_path = VECTORSTORE_PATHS[format_key]
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return FAISS(index=index, embedding_function=None, docstore=None, index_to_docstore_id=None)
    return None

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

@app.route('/')
def index():
    # Clear chat history when returning to the menu
    session.pop('chat_history', None)  # Clear chat history stored in session
    return render_template('index.html')

async def process_documents_async(pdf_docs, format_key):
    global vectorstore, conversation_chain
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks, format_key)
    conversation_chain = get_conversation_chain(vectorstore)

@app.route('/process', methods=['POST'])
def process_documents():
    pdf_docs = request.files.getlist('pdf_docs')
    format_key = request.form['format_key']
    asyncio.run(process_documents_async(pdf_docs, format_key))
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain

    # Retrieve chat history from session
    chat_history = session.get('chat_history', [])

    if request.method == 'POST':
        if conversation_chain is None:
            return redirect('/')  # Redirect to process documents first
        user_question = request.form['user_question']
        
        # Get the response from the conversation chain
        response = conversation_chain({'question': user_question})

        # Debugging: Log the response to see its structure
        print("Response from conversation chain:", response)

        # Update this part based on the actual structure of the response
        bot_response = response.get('response') or response.get('answer') or 'No response found'
        
        # Add to chat history
        chat_history.append({'user': user_question, 'bot': bot_response})
        
        # Save the chat history in the session
        session['chat_history'] = chat_history

    return render_template('chat.html', chat_history=chat_history)

@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history():
    # Clear the chat history stored in the session
    session['chat_history'] = []
    return redirect('/chat')

# Load the vectorstore when the app starts (for a default format)
vectorstore = load_vectorstore('format_a')
if vectorstore is not None:
    conversation_chain = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    app.run(debug=True)
