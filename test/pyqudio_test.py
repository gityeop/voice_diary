import pyaudio
import wave
import threading

# 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# 녹음 플래그
recording = True

# 녹음된 프레임을 저장할 리스트
frames = []

# PyAudio 객체 생성
audio = pyaudio.PyAudio()

# 녹음 함수
def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("녹음을 시작합니다. 중지하려면 'q'를 입력하세요.")
    
    global recording
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()

# 녹음 스레드 시작
thread = threading.Thread(target=record_audio)
thread.start()

# 사용자 입력 대기
while True:
    if input().lower() == 'q':
        recording = False
        break

# 녹음 스레드가 끝날 때까지 대기
thread.join()

# PyAudio 객체 종료
audio.terminate()

# WAV 파일로 저장
wf = wave.open("output.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print("녹음이 완료되었습니다. 'output.wav' 파일을 확인하세요.")