from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os

openai_api_key = ""
os.environ["OPENAI_API_KEY"] = openai_api_key

app = Flask(__name__)

# set allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

# function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/process_input", methods=["POST"])
def process_input():
    # get user input and uploaded file
    user = request.form.get("question")
    uploaded_file = request.files['file']

    # check if file is uploaded and has allowed extension
    if uploaded_file and allowed_file(uploaded_file.filename):
        # save file in current directory
        filename = secure_filename(uploaded_file.filename)
        uploaded_file.save(filename)

        # split text from file
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(filename)

        # create vector store from text
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings)

        # run query and get response
        qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectorstore)
        query = user
        res = qa.run(query)

        # remove uploaded file after use
        os.remove(filename)

        return render_template('index.html', response=res)
    else:
        return "Error: Please upload a PDF file."  # error message for invalid file

@app.route("/clear", methods=["POST"])
def clear():
    return render_template('index.html')  # return index page to clear input and response

if __name__ == '__main__':
    app.run(debug=True)
