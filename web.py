# app.py
from flask import Flask, render_template, request
import subprocess
import sys
import cv2,os
import requests
app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/haris/OneDrive/Desktop/bbbbb/test'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    if request.method == 'POST':
        # Get the path to the Python interpreter
        python_executable = sys.executable
        if 'video_file' not in request.files:
            return 'No file part'

        file = request.files['video_file']

        # If the user does not select a file, the browser may submit an empty file without a filename
        if file.filename == '':
            return 'No selected file'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        # Retrieve the filename
        filename = file.filename
        print(filename)
        # Get the path to the Python script you want to execute
        script_path = 'recognise_human_activity.py'  # Replace this with the actual path to your script
        # Run the Python script
        result = subprocess.run([python_executable, script_path,filename], capture_output=True, text=True)
        # Return the output of the script to the HTML page
        return result.stdout
if __name__ == '__main__':
    app.run(debug=True)

   
