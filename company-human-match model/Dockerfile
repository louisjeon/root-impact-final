FROM python:3.8-slim

WORKDIR /app
RUN python -m pip install --upgrade pip

# requirements.txt를 복사 후 의존성 설치
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# 애플리케이션 코드 전체 복사
COPY . /app

# Gunicorn을 사용하여 애플리케이션 실행 (포트 5000)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
