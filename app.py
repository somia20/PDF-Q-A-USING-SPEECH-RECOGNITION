# app.py
from multiprocessing import Process, Queue
import pyaudio
import wave
from faster_whisper import WhisperModel
from QA import answer_question

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

    queue.put('output.wav')


def start_recording(audio_queue):
    producer_process = Process(target=audio_producer, args=(audio_queue,))
    producer_process.start()
    producer_process.join()  # Wait for the recording to finish
    audio_path = audio_queue.get()  # Get the path of the recorded audio

    # Transcribe audio
    segments, _ = model.transcribe(audio_path, beam_size=5)
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text

        # Print the transcribed text
    print("Transcribed Text:", transcribed_text)

    # Specify the path to your PDF file
    pdf_path = "c:/Users/User/Desktop/AI.pdf"

    # Use transcribed text as the question to retrieve the answer from the PDF
    answer = answer_question(pdf_path, transcribed_text)
    print("Answer from PDF:", answer)


def stop_recording():
    global producer_process, consumer_process
    if producer_process:
        producer_process.terminate()
    else:
        print("Recording process is not running.")


if __name__ == "__main__":
    audio_queue = Queue()

    producer_process = Process(target=audio_producer, args=(audio_queue,))
    producer_process.start()

    start_recording(audio_queue)

    producer_process.join()
