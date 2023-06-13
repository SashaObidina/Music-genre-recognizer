import os

import requests
from common import queryMessage
import librosa #Python package for music & audio files
import librosa.display

# checking the file path for correctness
def check_correct(filepath):
    try:
        with open(filepath, encoding='utf-8') as file:
            if filepath.lower().endswith(('.wav')):
                print("Filepath is correct. \n")
                return True
            else:
                print("File must have an extension .wav, try again. \n")
                return False
    except FileNotFoundError:
        print(f"File {filepath} isn't found, try again. \n")
        return False

# rest service config
url = "http://127.0.0.1:8000/music/"
headers = {'Content-Type': 'application/json', 'Process-Id': f"{os.getpid()}"}

# inference; running the model
while True:
    # input interface
    user_input = input("Enter audio-file path in command prompt or 'q' to quit. Audio-file must have an extension .wav: \n")
    if user_input == 'q':
        break
    if not check_correct(user_input):
        continue

    y, s = librosa.load(user_input)
    y = y.tolist()

    # calling the model
    http_res = requests.post(url=url, headers=headers, data=queryMessage(time_series=y, sr=s).json())
    print(http_res)
    y = http_res.json()

    # output interface
    print(y)
