import requests
from datetime import datetime
import xmltodict
import asyncio
import time
import os
os.environ["SUNO_ENABLE_MPS"] = "True"
import librosa
import speech_recognition as sr
from funasr import AutoModel
import soundfile as sf
from scipy.io.wavfile import write as write_wav
from transformers import AutoProcessor, BarkModel
from pydub import AudioSegment
import wave
import pyaudio
def initialize_vad_model():
    print("Initializing VAD model...")
    model_dir = "iic/SenseVoiceSmall"
    vad_model = AutoModel(model=model_dir, vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 60000, "max_end_silence_time":2500}, device="cuda:0")
    return vad_model

def play_start_signal():
    # beep.mp3 파일을 불러오기
    signal = AudioSegment.from_file("assets/audio/beep.mp3", format="mp3")
    
    # pyaudio 설정
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=signal.channels,
                    rate=signal.frame_rate,
                    output=True)
    
    # 오디오 데이터를 재생
    stream.write(signal.raw_data)
    
    # 스트림 종료
    stream.stop_stream()
    stream.close()
    p.terminate()

def record_audio(vad_model):
    print("Recording audio...")
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=22050) as source:
        print("Please say something...")
        # Play start signal
        play_start_signal()
        audio_data = recognizer.listen(source)
        with open("temp/user_audio/temp.wav", "wb") as temp_file:
            temp_file.write(audio_data.get_wav_data())

    # Load and process the audio file
    input_file = "temp/user_audio/temp.wav"
    audio, sample_rate = sf.read(input_file)
    audio = audio.astype('float32')
    print("Running VAD on the recorded audio...")
    res = vad_model.generate(input=audio, fs=sample_rate)
    
    print(f"Detected VAD segments: {res}")

    # If VAD detected segments, save them
    if res and res[0]["text"]:
        transcription = res[0]["text"]  # Get transcription text
        segments = res[0].get("value", [])  # Get segments if available
        print(f"Segments: {segments}")
        if segments:
            segment_audio = AudioSegment.from_file(input_file)
            detected_audio = segment_audio[segments[0][0]:segments[0][1]]
            detected_audio.export(input_file, format="wav")
        else:
            print("No segments detected, only transcription is available.")
        return transcription  # Return transcription text
    else:
        print("No speech detected.")
        return None

async def convert_opus_to_wav(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000, res_type='kaiser_fast')
    sf.write(output_path, y, sr)


async def play_audio(file_path):


    # 오디오 파일 열기
    wf = wave.open(file_path, 'rb')

    # PyAudio 객체 생성
    p = pyaudio.PyAudio()

    # 오디오 스트림 열기
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 오디오 파일 읽고 재생
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # 스트림 종료
    stream.stop_stream()
    stream.close()

    # PyAudio 종료
    p.terminate()
    
def get_current_date_string():
    current_date = datetime.now().date()
    # %A를 사용하여 전체 요일 이름을 출력
    day_of_week = current_date.strftime("%A")
    return current_date.strftime("%Y%m%d") + f" ({day_of_week})"

def get_current_hour_string():
    now = datetime.now()
    if now.minute<45: # base_time와 base_date 구하는 함수
        if now.hour==0:
            base_time = "2330"
        else:
            pre_hour = now.hour-1
            if pre_hour<10:
                base_time = "0" + str(pre_hour) + "30"
            else:
                base_time = str(pre_hour) + "30"
    else:
        if now.hour < 10:
            base_time = "0" + str(now.hour) + "30"
        else:
            base_time = str(now.hour) + "30"

    return base_time
def forecast(url='http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst', keys=None, grid_x=None, grid_y=None):
    # 값 요청 (웹 브라우저 서버에서 요청 - url주소와 파라미터)
    params ={'serviceKey' : keys, 
        'pageNo' : '1', 
        'numOfRows' : '1000', 
        'dataType' : 'XML', 
        'base_date' : get_current_date_string(), 
        'base_time' : get_current_hour_string(), 
        'nx' : grid_x, 
        'ny' : grid_y }
    res = requests.get(url, params = params)

    #XML -> 딕셔너리
    xml_data = res.text
    dict_data = xmltodict.parse(xml_data)
    #값 가져오기
    weather_data = dict()
    for item in dict_data['response']['body']['items']['item']:
        # 기온
        if item['category'] == 'T1H':
            weather_data['tmp'] = item['fcstValue']
        # 습도
        if item['category'] == 'REH':
            weather_data['hum'] = item['fcstValue']
        # 하늘상태: 맑음(1) 구름많은(3) 흐림(4)
        if item['category'] == 'SKY':
            weather_data['sky'] = item['fcstValue']
        # 강수형태: 없음(0), 비(1), 비/눈(2), 눈(3), 빗방울(5), 빗방울눈날림(6), 눈날림(7)
        if item['category'] == 'PTY':
            weather_data['sky2'] = item['fcstValue']

    return weather_data

def proc_weather(address, dict_sky):
    str_sky = f"{address} "
    if dict_sky['sky'] != None or dict_sky['sky2'] != None:
        str_sky = str_sky + "날씨 : "
        if dict_sky['sky2'] == '0':
            if dict_sky['sky'] == '1':
                str_sky = str_sky + "맑음"
            elif dict_sky['sky'] == '3':
                str_sky = str_sky + "구름많음"
            elif dict_sky['sky'] == '4':
                str_sky = str_sky + "흐림"
        elif dict_sky['sky2'] == '1':
            str_sky = str_sky + "비"
        elif dict_sky['sky2'] == '2':
            str_sky = str_sky + "비와 눈"
        elif dict_sky['sky2'] == '3':
            str_sky = str_sky + "눈"
        elif dict_sky['sky2'] == '5':
            str_sky = str_sky + "빗방울이 떨어짐"
        elif dict_sky['sky2'] == '6':
            str_sky = str_sky + "빗방울과 눈이 날림"
        elif dict_sky['sky2'] == '7':
            str_sky = str_sky + "눈이 날림"
        str_sky = str_sky + "\n"
    if dict_sky['tmp'] != None:
        str_sky = str_sky + "온도 : " + dict_sky['tmp'] + 'ºC \n'
    if dict_sky['hum'] != None:
        str_sky = str_sky + "습도 : " + dict_sky['hum'] + '%'

    return str_sky
  
def get_grid_values(address, df):
    try:
        row = df.loc[df['3단계'] == address]
        if not row.empty:
            grid_x = row['격자 X'].values[0]
            grid_y = row['격자 Y'].values[0]
            return grid_x, grid_y
        else:
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None
