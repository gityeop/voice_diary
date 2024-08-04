from pydub import AudioSegment

def cut_audio(input_path, output_path, start_time, end_time):
    try:
        # 오디오 파일 로드
        audio = AudioSegment.from_mp3(input_path)
        
        # 오디오 정보 출력
        print(f"Duration: {len(audio) / 1000} seconds")  # 밀리초 단위를 초 단위로 변환
        
        # 자르기
        cut_audio = audio[start_time * 1000:end_time * 1000]  # pydub는 밀리초 단위 사용
        
        # 결과 저장
        cut_audio.export(output_path, format="mp3")
        
        print(f"Successfully cut the audio from {start_time} to {end_time} seconds.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 예시 사용법
input_path = "beep_sound.mp3"  # 올바른 경로 지정
output_path = "output_sound.mp3"  # 올바른 경로 지정
start_time = 1  # 시작 시간 (초)
end_time = 3  # 종료 시간 (초)

cut_audio(input_path, output_path, start_time, end_time)