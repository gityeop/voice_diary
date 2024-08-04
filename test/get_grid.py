import pandas as pd
import sys
import os

# 현재 파일의 경로를 가져와서 상위 디렉토리를 sys.path에 추가
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)

# 부모 디렉토리에 있는 utils 모듈에서 함수 가져오기
from utils import forecast, get_grid_values, proc_weather

# 데이터 예제 (CSV 파일 경로를 수정하세요)
data_file = os.path.join(parent_dir, 'data', 'gridx_y.csv')
df = pd.read_csv(data_file)
weather_api_key = os.getenv('WEATHER_API_KEY')

# 함수 사용 예제
keys = weather_api_key
url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst'
address = '화양동'
grid_x, grid_y = get_grid_values(address, df)

weather_data = forecast(url, keys, grid_x, grid_y)
print(proc_weather(address, weather_data))