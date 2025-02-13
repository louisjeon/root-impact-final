FROM python:3.8-slim

WORKDIR /app

# 시스템 패키지 업데이트 및 Tesseract OCR과 한국어 언어 데이터 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-kor \
    libtesseract-dev \
    libleptonica-dev \
    libgl1-mesa-glx \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# requirements.txt 복사와 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 애플리케이션 전체 코드 복사
COPY . .

# 컨테이너가 사용하는 포트 명시
EXPOSE 5000

# Gunicorn으로 Flask 애플리케이션 실행
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
