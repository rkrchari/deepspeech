import os
import logging
import faiss
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, session
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from dotenv import load_dotenv
from jinja2 import Template

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session

# Set up the upload folder and allowed file types
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Set up logging
logging.basicConfig(level=logging.INFO)

# FAISS Paths (for storing the vector index)
VECTORSTORE_PATH = 'faiss_index.index'

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def load_vectorstore():
    """
    Load an existing vector store from file if available.
    """
    if os.path.exists(VECTORSTORE_PATH):
        index = faiss.read_index(VECTORSTORE_PATH)
        return FAISS(index=index)
    return None

import json
from jinja2 import Template

def generate_test_cases_with_llm(requirements_text, requirement_type): 
    """
    Generate test cases using a language model based on the requirement type.
    """
    example_prompts = """
    Example 1: User Authentication System
    Test Case ID: TC_001
    Test Case Title: User Registration with Valid Email and Password
    Pre-Conditions: The user is on the registration page and has valid email and password credentials.
    Test Steps: 
        1. Enter a valid email address.
        2. Enter a valid password.
        3. Click on the "Register" button.
    Description: Ensure that users can successfully register with a valid email address and password.
    Expected Outcome: User should be registered and redirected to the login page.
    Pass/Fail Criteria: Pass if the user is registered and redirected; Fail if there is any error in the registration process.
    
    Example 2: Shopping Cart System
    Test Case ID: TC_002
    Test Case Title: Add Product to Cart
    Pre-Conditions: The user is logged in and browsing products.
    Test Steps:
        1. Navigate to the product details page.
        2. Click on the "Add to Cart" button.
        3. View the shopping cart.
    Description: Ensure that a product is successfully added to the user's shopping cart.
    Expected Outcome: The selected product should appear in the cart with the correct price and quantity.
    Pass/Fail Criteria: Pass if the product appears in the cart with correct details; Fail if the product is not added correctly.
    
    Example 3: Payment Gateway Integration
    Test Case ID: TC_003
    Test Case Title: Successful Payment Transaction
    Pre-Conditions: The user has items in their shopping cart and is on the checkout page.
    Test Steps:
        1. Enter valid credit card information.
        2. Click on the "Pay Now" button.
    Description: Ensure that the payment gateway processes the payment successfully.
    Expected Outcome: The payment should be processed, and the user should receive a confirmation message.
    Pass/Fail Criteria: Pass if the payment is processed successfully; Fail if any error occurs during the payment.
    """

    try:
        # Construct the prompt for LLM, including the examples and additional instructions
        prompt = f"""
        {example_prompts}
        
        Instructions:
        1. Only include meaningful descriptions. All extraneous details, values like '|', and irrelevant logic should be omitted.
        2. Ensure the descriptions focus on testing the {requirement_type} as part of the software SDLC cycle. Descriptions should be actionable and meaningful.
        3. Test cases should closely align with both user and business requirements. They should explicitly test the functional and non-functional aspects of the system.
        4. Test cases should consider **positive** and **negative scenarios** (e.g., valid and invalid inputs, edge cases, boundary conditions).
        5. Include test cases for **error handling** and **exception scenarios**. For instance, how the system behaves when invalid data is entered or a network failure occurs.
        6. Ensure that all **business rules** are tested, such as validation checks, price calculations, and regulatory compliance.
        7. Ensure test cases are **reusable** and **scalable**: they should be written in a way that they can be applied across different modules or use cases.
        8. Pay attention to performance-related scenarios like **load times**, **scalability**, and **system resource usage** when relevant.
        9. Consider **security testing**: Test for vulnerabilities like **SQL injection**, **cross-site scripting (XSS)**, and **data leaks** when relevant.
        10. Test for **usability**: Test how intuitive the system is for end-users, ensuring it is easy to navigate and interact with.
        11. Ensure the test cases cover **user roles**: e.g., Admin, User, Guest. Test different permissions, visibility, and accessibility for each role.
        12. Add **boundary tests** for fields such as email address length, password length, credit card number length, and other input data lengths.
        13. Ensure that test cases are clear and precise, with all actions and expected results written in **user-friendly language**.
        14. Pay attention to **multi-device and multi-browser compatibility** if the system is web or mobile-based.
        15. Add **localization and internationalization** considerations when testing, such as different languages, currencies, or time zones.
        16. Ensure that the **test steps** are detailed enough for testers to execute the case without ambiguity, and **expected outcomes** are clear and measurable.
        17. Ensure that each test case includes a **Description** field with a clear, concise explanation of the test case's purpose.
        18. The **Description** should describe the objective of the test case, the functionality being tested, and why it's important. 
        19. Need to have Test Case ID, Test Case Title, Pre-Conditions,Test Steps, Description,Expected Outcome,Pass/Fail Criteria in the model response output

        
        {requirement_type}:
        {requirements_text}
        
        Now, based on the above {requirement_type}, generate a set of detailed test cases, focusing on the business logic, conditions, and actions that should be tested.
        """

        # Make the API call to generate test cases using Google Gemini or other API.
        model = genai.GenerativeModel(model_name='gemini-1.5-flash-002')
        response = model.generate_content(prompt)

        # Extract test case content from the LLM response
        try:
            test_case_text = response.text
            logging.info(f"Extracted test case text: {test_case_text}")
        except KeyError as e:
            logging.error(f"Error extracting test case text: {e}")
            return json.dumps([])

        # If the response text is empty or not valid, return an empty list
        if not test_case_text.strip():
            logging.error("Generated test case content is empty.")
            return json.dumps([])

        # Clean the response to make the list of test cases readable
        import re
        cleaned_test_cases = [
            re.sub(r"(Test\s*Case)\s*\d+", r"\1", line.strip())  # Removes number after 'Test Case'
            .replace("'", '')  # Remove single quotes from the text
            .replace('*', '')  # Remove asterisks
            .replace('**', '')  # Remove double asterisks
            for line in test_case_text.strip().split('\n') if line.strip()
        ]

        # Strip any extra quotes and spaces from the cleaned list
        cleaned_test_cases1 = [item.strip("'") for item in cleaned_test_cases]

        # Initialize the result list to store test cases
        result = []
        test_case = {}

        inside_test_steps = False  # New flag to track when we're inside test steps

        for item in cleaned_test_cases1:
            item = item.strip()
            if item.startswith('Test Case ID'):
                test_case['Test Case ID'] = re.search(r':(.*)', item).group(1)
                test_case_id = test_case["Test Case ID"]
                inside_test_steps = False  # Reset flag for new test case
            elif item.startswith('Test Case Title'):
                test_case['Test Case Title'] = re.search(r':(.*)', item).group(1)
                test_case_title = test_case["Test Case Title"]
            elif item.startswith('Pre-Conditions'):
                test_case['Pre-Conditions'] = re.search(r':(.*)', item).group(1)
                description = test_case['Pre-Conditions']
            elif item.startswith('Test Steps'):
                # Start collecting test steps
                inside_test_steps = True
                test_case['Test Steps'] = ""  # Initialize empty string for test steps
            elif item.startswith('Description'):
                test_case['Description'] = re.search(r':(.*)', item).group(1)
                description = test_case['Description']
            elif item.startswith('Expected Outcome'):
                test_case['Expected Outcome'] = re.search(r':(.*)', item).group(1)
                expected_outcome = test_case['Expected Outcome']
                result.append(test_case.copy())  # Append the test case to result
                test_case = {}  # Reset the dictionary for the next test case
                inside_test_steps = False  # End test steps collection
            elif inside_test_steps:  # If inside test steps, keep adding to 'Test Steps'
                if test_case.get('Test Steps') is not None:
                    test_case['Test Steps'] += "\n" + item
            else:
                logging.info(f"Skipping item: {item}")

        logging.info(f"Generated test cases: {result}")
        return json.dumps(result)

    except Exception as e:
        logging.error(f"Error generating test cases: {str(e)}")
        return json.dumps([{
            "test_case_id": "TC_001",
            "test_case_title": "Default Test Case",
            "description": f"An error occurred while generating test cases: {str(e)}",
            "expected_output": "Test Case generation failure"
        }])



def save_test_cases_as_html(test_cases_json):
    """
    Save the test cases from JSON string to an HTML file using a Jinja2 template.
    """
    try:
        # Parse the test cases JSON into Python objects
        test_cases = json.loads(test_cases_json)

        if not test_cases:
            logging.error("No test cases to generate HTML.")
            return None

        # Define the HTML template for rendering the test cases
        template_str = """
        <html>
        <head>
            <title>Generated Test Cases</title>
            <style>
                body { font-family: Arial, sans-serif; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Generated Test Cases</h1>
            <table>
                <tr><th>Test Case ID</th><th>Test Case Title</th><th>Pre-Conditions</th></th><th>Test Steps</th><th>Description</th><th>Expected Outcome</th></th><th>Pass/Fail Criteria</th></tr>
                {% for test_case in test_cases %}
                    <tr>
                        <td>{{ test_case['Test Case ID']}}</td>
                        <td>{{ test_case['Test Case Title']}}</td>
                 
                        <td>{{ test_case['Pre-Conditions']}}</td>
                        <td>{{ test_case['Test Steps']}}</td>
                        <td>{{ test_case['Description']}}</td>
                        <td>{{ test_case['Expected Outcome']}}</td>
                        <td>{{ test_case['Pass/Fail Criteria']}}</td>
                    </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """

        # Use Jinja2 template to generate HTML
        template = Template(template_str)
        html_output = template.render(test_cases=test_cases)

        output_file = 'generated_test_cases.html'
        with open(output_file, 'w') as file:
            file.write(html_output)

        logging.info(f"HTML file generated at {output_file}")
        return output_file

    except Exception as e:
        # Log any exceptions while saving the HTML file
        logging.error(f"Error saving test cases as HTML: {str(e)}")
        return None


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'business_req_doc' not in request.files or 'requirement_type' not in request.form:
        return redirect('/')

    file = request.files['business_req_doc']
    requirement_type = request.form['requirement_type']  # Get the selected requirement type

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract text from the uploaded PDF
        raw_text = get_pdf_text([file])

        # Split text into chunks
        text_chunks = get_text_chunks(raw_text)

        # Create or load the vector store
        vectorstore = load_vectorstore()
        if vectorstore is None:
            vectorstore = get_vectorstore(text_chunks)

        # Generate test cases using the LLM with the selected requirement type
        test_cases = generate_test_cases_with_llm(raw_text, requirement_type)

        if not test_cases or test_cases == "[]":
            logging.error("No valid test cases generated.")
            return redirect('/')  # Redirect if no test cases are generated

        # Save test cases as HTML
        html_file_path = save_test_cases_as_html(test_cases)

        if not html_file_path:
            logging.error("Failed to generate HTML file for test cases.")
            return redirect('/')  # Redirect if HTML generation fails

        # Read the HTML content to render in the page
        with open(html_file_path, 'r') as file:
            html_content = file.read()

        # Set the header text based on the requirement type
        page_header = f"Test Case for {requirement_type}"

        # Render HTML content in the panel and allow download
        #return render_template('index.html', html_content=html_content, html_file_path=html_file_path, page_header=page_header)
        return render_template('index.html', html_content=html_content, html_file_path=html_file_path, page_header=page_header, requirement_type=requirement_type)


    return redirect('/')




from flask import send_file

@app.route('/download/<filename>')
def download_html(filename):
    return send_file(filename, as_attachment=True)


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # Retrieve chat history from session
    chat_history = session.get('chat_history', [])

    if request.method == 'POST':
        # Check if vectorstore exists
        vectorstore = load_vectorstore()
        if vectorstore is None:
            return redirect('/')  # Redirect to upload page if no documents are processed

        # Retrieve the question asked by the user
        user_question = request.form['user_question']

        # Get the conversation chain for the chat
        conversation_chain = get_conversation_chain(vectorstore)

        # Get the response from the conversation chain
        response = conversation_chain({'question': user_question})

        # Extract the response text
        bot_response = response.get('response') or response.get('answer') or 'No response found'

        # Add to chat history
        chat_history.append({'user': user_question, 'bot': bot_response})

        # Save the chat history in the session
        session['chat_history'] = chat_history

    return render_template('chat.html', chat_history=chat_history)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history():
    # Clear the chat history stored in the session
    session['chat_history'] = []
    return redirect('/chat')

if __name__ == '__main__':
    app.run(debug=True)
