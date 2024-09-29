from flask import Flask, render_template, request, jsonify
from rag_chatgpt import RagModel
import os

upload_folder = 'uploads'

if not(os.path.exists(upload_folder)):
    os.mkdir(upload_folder)

model = RagModel()

try:
    model.load_embeddings()
    print("Succesfully loaded embeddings on startup")
except:
    print("Unable to load embeddings")

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = upload_folder

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Frontend (chat interface)
@app.route('/')
def chat():
    return render_template('chat.html')

# Send Message
@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form['message'].lower()

    response, references = model.ask(user_message, return_answer_only=False)

    response = response + "\n\n<strong>References(by relevance)</strong>\n" + references

    return jsonify({"response": response})

# Admin
@app.route('/admin')
def admin():
    return render_template('admin.html')

# File Upload
@app.route("/upload", methods=['POST'])
def upload_file_post():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return f"File {filename} uploaded successfully!", 200
    else:
        return 'File type not allowed', 400


# File Processing
@app.route('/process_files', methods=['POST'])
def process_files():
    
    model.process_files()
    model.load_embeddings()
    return jsonify({"status": "Files processed successfully!"})


if __name__ == '__main__':
    app.run(debug=True)