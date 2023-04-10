# whisper_ing

### To enable seemingly __real time__ process of audio transcription using WhisperAI.
```bash
# Environment variables (Preferably to be executed in venv)
* python 3.8 and above
* ffmpeg installed

# Clone the repository
git clone https://github.com/saioru/whisper_ing.git

# Install requirements
pip install -r requirements.txt

# Execute
python transcribe.py 
```

|      Argument      	|  Default Value 	|                                        Description                                        	|
|:------------------:	|:--------------:	|:-----------------------------------------------------------------------------------------:	|
| --model            	| Whisper (Base) 	| Supports downloaded models by file path, or standard Whisper models.                      	|
| --use_en           	|      True      	| Optional when given file path to downloaded model.                                        	|
| --energy_threshold 	|      1000      	| Energy level for mic to detect, audio levels lower than threshold are considered silence. 	|
| --sample_rate      	|      16000     	| Standard audio sampling rate in sync with Whisper architecture, CHANGE AT OWN RISK.       	|
| --transcript_delay 	|        2       	| Define (n) seconds of buffer delay in transcription.                                      	|
| --phrase_timeout   	|        3       	| Define (n) seconds to indicate the next input as next line.                               	|
