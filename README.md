**PDF-QA-USING-SPEECH-RECOGNITION**

**Overview :**

This repository contains code for a speech recognition and question answering system integrated with a PDF document. The system records audio input, transcribes it, retrieves relevant information from a PDF document, and generates answers to questions asked by the user.

**Contents:**

- Vector.py: It Defines functions to create and load vector storage for document embeddings using SentenceTransformer - and FAISS.
- QA.py: Implements functions to answer questions using document retrieval and text generation techniques.
main_script.py: Provides an example of how to use the answer_question function to ask a question and retrieve an answer from a PDF document.

**Workflow**

![alt text](<Blank diagram.png>)

**Requirements:**

To run the code in this repository, you need:
- Python 3.x
- Required Python packages: sentence_transformers, langchain_community, PyMuPDF, google, pyaudio, wave, faster_whisper, translate, time

**Usage:**

- Setting up the environment:
Install the required Python packages using pip:
pip install -r requirements.txt

- Running the main script:
Execute the main_script.py file to ask a question and retrieve an answer from a PDF document.
python main_script.py

- Speech Recognition and Question Answering:
Run the app.py file to start the speech recognition and question answering system. This will record audio input, transcribe it, and provide answers based on the content of the specified PDF document.
python app.py

**Additional Notes:**

- Ensure that the PDF document path is correctly specified in the code.
- Make sure to have a valid Google API key configured in the environment variables for the text generation functionality.
- Adjust the language translation settings as needed in the audio_consumer function in app.py.
- This system uses the SentenceTransformer library for document embeddings and the FAISS library for efficient similarity search.
- For any issues or improvements, feel free to raise an issue or submit a pull request.
