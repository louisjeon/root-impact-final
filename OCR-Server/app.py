from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, UnidentifiedImageError
from pdf2image import convert_from_bytes
import cv2
import numpy as np

app = Flask(__name__)

def preprocess_image(pil_image):
    # PIL 이미지 -> numpy 배열 변환
    image = np.array(pil_image)
    
    # 알파 채널 있는 경우 변환 (RGBA -> RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 컬러이미지인 경우 그레이스케일로 변환
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 잡음 제거를 위한 가우시안 블러 적용
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Otsu의 이진화로 명암 대비 강화
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 추가 전처리(예: 확대, 팽창/침식 등) 가능
    return Image.fromarray(image)

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400

    file = request.files['file']
    extracted_text = ""
    
    # OCR 호출 시 사용할 설정 값 (여기서는 PSM 모드 6 사용)
    config = '--psm 6'
    
    # 파일 확장자가 PDF인 경우
    if file.filename.lower().endswith('.pdf'):
        try:
            # PDF 전체 페이지를 이미지 리스트로 변환
            images = convert_from_bytes(file.read())
            if not images:
                return jsonify({"error": "PDF 파일에서 이미지를 찾을 수 없습니다."}), 400

            texts = []
            # 각 페이지에 대해 전처리와 OCR 적용
            for page_num, image in enumerate(images, start=1):
                proc_image = preprocess_image(image)
                text = pytesseract.image_to_string(proc_image, lang='kor', config=config)
                texts.append(f"--- Page {page_num} ---\n{text}")
            extracted_text = "\n".join(texts)
        except Exception as e:
            return jsonify({"error": f"PDF 변환 중 오류 발생: {str(e)}"}), 500
    else:
        try:
            # PDF가 아닌 경우 일반 이미지 처리
            img = Image.open(file.stream)
            proc_img = preprocess_image(img)
            extracted_text = pytesseract.image_to_string(proc_img, lang='kor', config=config)
        except UnidentifiedImageError:
            return jsonify({"error": "유효한 이미지 파일이 아닙니다."}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"text": extracted_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
