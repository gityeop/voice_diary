import os
from openai import OpenAI
import pyaudio
import wave
import numpy as np
from funasr import AutoModel
import torch
from utils import convert_opus_to_wav, record_audio, play_audio
from transformers import Wav2Vec2ForCTC,AutoProcessor 
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)
conversation_history = [
    {
        "role": "system",
        "content": "Speak in Korean. You are a diary-writing mentor. Follow the format below to ask questions one at a time. Asking additional questions is okay if they don't disrupt the natural flow. Once all checklist information is gathered, compile it into today's diary entry. After writing the diary, summarize tasks or important information for tomorrow at the end. Do not use informal language during the process. You must speak in Korean and ask one topic at a time. 루틴 체크 및 일기 체크리스트- 운동- 유연성- 어떤 유연성 운동- 근력운동- 어떤 근육 운동- 일에 대한 질문- 일이 어땠는지- 어떤 사건이 있었고- 사교적인 일에서 처리해야하는 일- 내가 나에게 해주고 싶은 말- 내일 했으면 하는 일, 해야하는 일(너가 해야하는 것)- 너가 오늘의 나에게 해줄 수 있는 말 + 오늘 하루 마무리 명언 추천"
    }
]
# 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1600  # 100ms
vad_kwargs = {
    "speech_noise_thres": 0.1     # 임계값
}
# VAD 모델 로드

# PyAudio 객체 생성
audio = pyaudio.PyAudio()

# 녹음된 프레임을 저장할 리스트
frames = []

# VAD 캐시
cache = {}

# 무음 감지를 위한 변수
silence_threshold = 15  # 1초의 무음
speech_threshold = 3

# 녹음 함수
vad_model = AutoModel(model="fsmn-vad", vad_kwargs=vad_kwargs)  # 기존 VAD 모델 초기화 코드
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h")
model_dir = "iic/SenseVoiceSmall"
model = AutoModel(model=model_dir, trust_remote_code=True, device="cuda:0")
def real_time_transcription(model, rate=16000, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("녹음을 시작하겠습니다. 말씀하세요.")
    frames = []
    recording = False
    silence_counter = 0
    speech_counter = 0
    speech_threshold = 3
    silence_threshold = 2
    try:
        while True:
            data = stream.read(chunk, exception_on_overflow=False)

            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)


            # VAD와 함께 모델 사용 (원하는 경우 VAD 모델을 별도로 통합 가능)
            res = model.generate(input=audio_chunk, cache={}, is_final=False, chunk_size=100)
            print(res)

            if res and res[0]['text']:
                text = res[0]['text']
                if not recording:
                    print("말하기 감지됨, 녹음 시작...")
                    recording = True
                frames.append(audio_chunk)
            else:
                if recording:
                    silence_counter += 1
                    recording = False
                else:
                    silence_counter += 1
                    if speech_counter > speech_threshold and silence_counter >= silence_threshold:
                        print("말하기가 끝난 것으로 감지됨, 녹음 중지.")
                        break

            if recording:
                # 실시간 transcription 수행
                audio_data = np.concatenate(frames, axis=0)
                input_values = processor(audio_data, return_tensors="pt", sampling_rate=rate).input_values
                logits = asr_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.decode(predicted_ids[0])
                print(f"Transcription: {transcription}")
                frames = []  # Reset frames for the next segment
    except KeyboardInterrupt:
            pass

    stream.stop_stream()
    stream.close()
    p.terminate()



while True:
  # 녹음 실행
    model_dir = "iic/SenseVoiceSmall"
    model = AutoModel(model=model_dir, trust_remote_code=True, device="cuda:0")
    real_time_transcription(model)
    

    # PyAudio 객체 종료
    audio.terminate()

    # WAV 파일로 저장
    if frames:
        wf = wave.open("output.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("녹음이 완료되었습니다. 'output.wav' 파일을 확인하세요.")
    else:
        print("녹음된 데이터가 없습니다.")
        
    if not os.path.exists('output.wav'):
        print("File 'test.wav' not found. Please ensure recording was successful.")
        continue

    audio_file = open('output.wav', 'rb')
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    ) # 한국말 Transcription 
    print(f"Transcription: {transcription.text}")

    user_message = {
        "role": "user",
        "content": transcription.text
    }

    conversation_history.append(user_message)

    print("Generating chat response...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history
    )

    bot_message = response.choices[0].message.content
    print(f"Chatbot response: {bot_message}")

    # Adding the bot response to the conversation history
    conversation_history.append({
        "role": "assistant",
        "content": bot_message
    })

    print("Converting chat response to speech...")
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=bot_message,
        response_format="wav",
    ) as tts_response:
        with open('tts.wav', 'wb') as audio_file:
            for chunk in tts_response.iter_bytes(chunk_size=1024):
                audio_file.write(chunk)
    print("Converted chat response to speech and saved to output.opus")

    # text_to_speech_bark(bot_message, output_file="output.wav")
    # convert_opus_to_wav('output.opus', 'output.wav')
    
    print("Playing audio response...")
    play_audio('tts.wav')
    print("Audio playback finished.")
    input("waiting for next recording...")