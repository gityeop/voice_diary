import librosa
import simpleaudio as sa
import numpy as np
import soundfile as sf

def load_opus_file(file_path):
    # .opus 파일 로드
    y, sr = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
    return y, sr

def save_as_wav(y, sr, output_path):
    # librosa로 로드한 데이터를 .wav 파일로 저장
    sf.write(output_path, y, sr)

def play_audio(file_path):
    print("Loading audio file for playback...")
    try:
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        print(f"Audio file loaded: {file_path}")
        print(f"Channels: {wave_obj.num_channels}, Sample Width: {wave_obj.bytes_per_sample}, Frame Rate: {wave_obj.sample_rate}")

        playback = wave_obj.play()
        print("Playback started...")
        playback.wait_done()
        print("Audio playback finished.")
    except Exception as e:
        print(f"Exception during playback: {e}")

def main():
    input_opus_path = 'output.opus'
    output_wav_path = 'output.wav'

    # .opus 파일 로드
    y, sr = load_opus_file(input_opus_path)
    print(f"Loaded .opus file with sample rate: {sr} and duration: {len(y) / sr} seconds")

    # .wav 파일로 저장
    save_as_wav(y, sr, output_wav_path)
    print(f"Saved .wav file to {output_wav_path}")

    # .wav 파일 재생
    play_audio(output_wav_path)

if __name__ == "__main__":
    main()
