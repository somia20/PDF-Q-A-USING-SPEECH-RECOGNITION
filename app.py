from multiprocessing import Process, Queue
import pyaudio
import wave
from faster_whisper import WhisperModel
from QA import answer_question
from translate import Translator
import time

model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

producer_process = None
consumer_process = None

def audio_producer(queue):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 10

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Producer: Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    audio_data = b''.join(frames)

    with wave.open('output.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)

    queue.put('output.wav')  # Put the WAV file path into the queue

def audio_consumer(queue):
    while True:
        if not queue.empty():
            audio_data = queue.get()
            print("Consumer: Transcribing...")
            segments, info = model.transcribe(audio_data, beam_size=5)
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            if info.language != "en":
                translator = Translator(to_lang="en", from_lang=info.language)
                transcribed_text = ""
                for segment in segments:
                    # Retry fetching the token up to 3 times
                    for _ in range(3):
                        try:
                            translated_text = translator.translate(segment.text)
                            transcribed_text += translated_text + " "
                            break  # Exit the retry loop if successful
                        except Exception as e:
                            print("Error fetching token:", e)
                            time.sleep(1)  # Wait for 1 second before retrying
                    else:
                        print("Failed to fetch token after retries.")
                print("Translated Text:", transcribed_text)
            else:
                transcribed_text = " ".join([segment.text for segment in segments])
                print("Transcribed Text:", transcribed_text)

            # Specify the path to your PDF file
            pdf_path = "c:/Users/User/Desktop/AI.pdf"

            # Retrieve the answer from the PDF using the transcribed text as the question
            answer = answer_question(pdf_path, transcribed_text)
            print("Answer from PDF:", answer)

def start_recording(audio_queue):
    producer_process = Process(target=audio_producer, args=(audio_queue,))
    producer_process.start()
    producer_process.join()  # Wait for the recording to finish
    audio_path = audio_queue.get()  # Get the path of the recorded audio

    consumer_process = Process(target=audio_consumer, args=(audio_queue,))
    consumer_process.start()

def stop_recording():
    global producer_process, consumer_process
    if producer_process:
        producer_process.terminate()
    else:
        print("Recording process is not running.")
    if consumer_process:
        consumer_process.terminate()
    else:
        print("Consumer process is not running.")

if __name__ == "__main__":
    audio_queue = Queue()

    producer_process = Process(target=audio_producer, args=(audio_queue,))
    producer_process.start()

    start_recording(audio_queue)

    producer_process.join()
