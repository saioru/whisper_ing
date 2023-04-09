# Reference https://github.com/davabase/whisper_real_time

import io
import os
import torch
import whisper
import argparse
import speech_recognition as sr
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from queue import Queue
from time import sleep

def load_model(model: str, use_en: bool = True):
    # Load / Download model
    if model != "large" and not use_en: model = model + ".en"
    try: model = whisper.load_model(model)
    except:
        model = whisper.load_model('base')
        print("Warning: Failed to load model, using default Whisper 'base' model") 
    print("--- Model loaded ---\n Transcription: <Speak to Start>\n")
    return model

def load_recognizer(energy_threshold: int, dynamic_energy_threshold: bool = False):
    # SpeechRecognizer to record audio for end speech detection.
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    # Dynamic energy compensation lowers the energy threshold dramtically that causes SpeechRecognizer to endlessly record.
    recorder.dynamic_energy_threshold = dynamic_energy_threshold
    return recorder

def load_input(sample_rate: int):
    # Define audio input source and sample rate.
    return sr.Microphone(sample_rate=sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", default="base", help="File path or abbreviation of a Whisper model", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--use_en", action="store_true", help="Enable English model")
    parser.add_argument("--energy_threshold", default=1000, help="Energy level for mic to detect.", type=int)   # Audio energy level
    parser.add_argument("--sample_rate", default=16000, help="Audio sample rate.", type=int)    # Audio sample rate
    parser.add_argument("--transcript_delay", "-d", default=2, help="When and during input, define (n) seconds of buffer delay in transcription.", type=float)
    parser.add_argument("--phrase_timeout", "-t", default=3, help="When no input detected, define (n) seconds to indicate the next input as next line.", type=float)  
    args = parser.parse_args()

    # The last time a recording was retreived from the queue, default None as initialization
    phrase_time = None
    # When and during input, define (n) seconds of buffer delay in transcription
    transcript_delay = args.transcript_delay 
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # When no input detected, define (n) seconds to indicate the next input as next line
    phrase_timeout = args.phrase_timeout

    # Initiate Speech Recognizer
    recorder = load_recognizer(args.energy_threshold)

    # Initiate Audio Input
    source = load_input(args.sample_rate)

    # Initialize Whisper transcription model
    model = load_model(args.model_path)

    # Reduce ambient noise
    with source: recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread to pass raw audio bytes using SpeechRecognizer
    recorder.listen_in_background(source, record_callback, phrase_time_limit=transcript_delay)

    # Define variables
    temp_file = NamedTemporaryFile().name   # temp file to store raw to wav data conversion
    transcription = ['']
    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue
            if not data_queue.empty():
                phrase_complete = False
                # Clear the current working audio buffer to start over with the new data
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True  # break off recording chunk
                # Refresh time when received new audio data from the queue
                phrase_time = now

                # Concat current audio data with the latest audio data
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                with open(temp_file, 'w+b') as f: f.write(wav_data.read())

                # Transcribe
                result = model.transcribe(temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # Add new item in transcript when pause detected
                if phrase_complete: transcription.append(text)
                else: transcription[-1] = text  # else edit the existing one

                # Clear console and reprint updated transcription
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription: print(line)
                print('', end='', flush=True) # flush stdout
                sleep(0.25) # sleep to prevent processor break
        except KeyboardInterrupt: break