from pydub import AudioSegment
from pydub.playback import play

audio_file = "song.wav"  # Change to your file
audio = AudioSegment.from_file(audio_file)
play(audio)
