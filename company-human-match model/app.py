from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# 사전 학습된 모델 불러오기
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_embedding(text):
    # 텍스트를 임베딩 벡터로 변환
    embedding = model.encode(text)
    # JSON 직렬화를 위해 리스트로 반환
    return embedding.tolist()

def calculate_cosine_similarity(emb1, emb2):
    # 두 임베딩 벡터 사이의 코사인 유사도 계산
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

@app.route('/embedding', methods=['POST'])
def embedding():
    # 클라이언트로부터 텍스트 데이터 수신
    input_text = request.json.get('text')
    embedding_vector = calculate_embedding(input_text)
    return jsonify({"embedding": embedding_vector})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
