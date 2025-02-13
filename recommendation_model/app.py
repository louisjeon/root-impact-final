from fastapi import FastAPI
import pandas as pd
from model import recommend_region_for_business

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Business Recommendation API is running!"}


@app.post("/recommend/")
def get_recommendation(data: dict):
    """
    기업 데이터를 입력받아 추천 지역 및 규제특구 정보를 반환하는 API.
    요청 데이터 예시:
    {
      "business_data": [
        {"기업명": "A기업", "산업군": "소매", "사업 설명": "패션업체", "투자 규모 (억원)": 18},
        {"기업명": "B기업", "산업군": "금융", "사업 설명": "대형 은행", "투자 규모 (억원)": 150}
      ]
    }
    """
    business_df = pd.DataFrame(data["business_data"])
    industry_df = pd.read_csv("data/industry_scores.csv", encoding="utf-8")
    regulatory_df = pd.read_csv("data/specialize_illgal.csv", encoding="utf-8")
    opportunity_df = pd.read_csv("data/opportunity.csv", encoding="utf-8")
    local_develop_df = pd.read_csv("data/local_develop.csv", encoding="utf-8")
    infra_df = pd.read_csv("data/Converted_Infra_Data.csv", encoding="utf-8")
    eco_df = pd.read_csv("data/eco111.csv",encoding="utf-8")

    result = recommend_region_for_business(business_df, industry_df, regulatory_df,opportunity_df,local_develop_df,infra_df,eco_df)
    return result.to_dict(orient="records")