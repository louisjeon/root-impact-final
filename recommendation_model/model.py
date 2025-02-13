import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import re

# BERT 모델 로드
model_name = "kykim/bert-kor-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)


# 시/도 단위 지역명 추출 함수
def extract_main_region(address):
    match = re.match(r'([^ ]+)', str(address))  # 첫 번째 공백 이전의 단어 추출
    return match.group(1) if match else ""

# 투자 규모에 따른 기업 유형 분류
def classify_business_size(investment):
    if investment < 10:
        return '소상공인'
    elif investment < 120:
        return '소기업'
    elif investment < 300:
        return '중기업'
    else:
        return '대기업'


# 기업 유형별 최고 점수 지역 찾기
def get_top_industries_by_business_size(df, business_size):
    if business_size not in df.columns:
        return pd.DataFrame(columns=["지역별", "산업별(10차)대분류", business_size])
    industry_max_scores = df.groupby("산업별(10차)대분류")[business_size].transform(max)
    top_regions = df[df[business_size] == industry_max_scores]

    return top_regions.groupby("산업별(10차)대분류").apply(lambda x: x.sample(1)).reset_index(drop=True)


# BERT 임베딩 추출 (사업 설명을 함께 반영)
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # 배치 차원만 제거하도록 squeeze(0)를 사용하여 (1, hidden_dim) -> (hidden_dim,)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()


def get_most_similar_region(region, infra_df, tokenizer, bert_model):
    """
    가장 유사한 지역을 BERT 임베딩 유사도로 찾는 함수

    Parameters:
        region (str): 조회할 지역명
        infra_df (pd.DataFrame): 인프라 데이터프레임
        tokenizer: BERT 토크나이저 객체
        bert_model: BERT 모델 객체

    Returns:
        str: 가장 유사한 지역명
    """
    region_vector = get_embedding(region).reshape(1, -1)
    best_match = None
    best_score = -1

    for candidate_region in infra_df["지역"].unique():
        candidate_vector = get_embedding(candidate_region).reshape(1, -1)
        score = cosine_similarity(region_vector, candidate_vector)[0][0]
        if score > best_score:
            best_match = candidate_region
            best_score = score

    return best_match if best_match else region
# BERT 기반 거리 추천 (산업군 + 사업 설명 반영)
def recommend_specialized_street(region, industry, business_desc, specialized_street_data):
    relevant_streets = specialized_street_data[
        specialized_street_data["대지역"].astype(str).str.contains(region, na=False, case=False)]

    if relevant_streets.empty:
        return "추천 거리 없음"

    # 거리 설명을 BERT 임베딩 변환
    street_descriptions = relevant_streets["거리소개"].astype(str).tolist()
    street_embeddings = np.array([get_embedding(desc) for desc in street_descriptions]).squeeze()

    # 산업군 + 사업 설명을 결합한 임베딩 생성
    combined_text = f"{industry} {business_desc}"
    industry_embedding = get_embedding(combined_text).squeeze()

    # 코사인 유사도 계산
    similarities = cosine_similarity([industry_embedding], street_embeddings).flatten()
    best_match_index = similarities.argmax()
    best_similarity_score = similarities[best_match_index]
    print(similarities[best_match_index])
    if best_similarity_score >= 0.2:
        return relevant_streets.iloc[best_match_index][["거리명", "소재지도로명"]].to_dict()
    return "추천 거리 없음"

# BERT 기반 규제특구 유사도 분석 함수
def find_best_matching_regulatory_zone(business_desc, regulatory_df):
    business_embedding = get_embedding(business_desc)
    regulatory_df["통합설명"] = (
            regulatory_df["사업명"] + " " +
            regulatory_df["현행 규제내용"] + " " +
            regulatory_df["완화된 규제내용"]
    )
    regulatory_texts = regulatory_df["통합설명"].astype(str).tolist()
    regulatory_embeddings = [get_embedding(text) for text in regulatory_texts]

    similarities = cosine_similarity([business_embedding], regulatory_embeddings).flatten()
    # 유사도가 0.2 이상인 인덱스 모두 선택
    indices = [idx for idx, score in enumerate(similarities) if score >= 0.2]

    if not indices:
        return None

    # 유사도 순으로 내림차순 정렬
    indices = sorted(indices, key=lambda i: similarities[i], reverse=True)

    results = []
    for idx in indices:
        row = regulatory_df.iloc[idx]
        results.append({
            "지역명": row["지역"],
            "특구명": row["사업명"],
            "현행 규제내용": row["현행 규제내용"],
            "완화된 규제내용": row["완화된 규제내용"],
            "유사도 점수": round(float(similarities[idx]), 4)
        })

    return results
# BERT 기반 규제특구 유사도 분석 함수
def find_best_matching_opportunity_zone(business_desc, opportunity_df):
    business_embedding = get_embedding(business_desc)
    print("----------")
    print(opportunity_df)
    opportunity_texts = opportunity_df["세부사업"].astype(str).tolist()
    opportunity_embeddings = [get_embedding(text) for text in opportunity_texts]

    similarities = cosine_similarity([business_embedding], opportunity_embeddings).flatten()
    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    if best_score >= 0.2:
        best_match = opportunity_df.iloc[best_match_index]
        return {
            "지역명": best_match["지역"],
            "기회발전특구사업": best_match["세부사업"],
            "유사도 점수": round(float(best_score), 4)
        }
    return None
def find_best_matching_develop_zone(business_desc, local_develop_df):
    business_embedding = get_embedding(business_desc)
    local_develop_df["통합설명"] = (
            local_develop_df["지역"] + " " +
            local_develop_df["특구명"] + " " +
            local_develop_df["혜택"]
    )
    print("----------")
    print(local_develop_df)
    local_texts = local_develop_df["통합설명"].astype(str).tolist()
    local_embeddings = [get_embedding(text) for text in local_texts]

    similarities = cosine_similarity([business_embedding], local_embeddings).flatten()
    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    if best_score >= 0.5:
        best_match = local_develop_df.iloc[best_match_index]
        print("--------------")
        print(best_match)
        return {
            "지역명": best_match["지역"],
            "지역개발특구": best_match["특구명"],
            "x": best_match["위도"],
            "y": best_match["경도"],
            "혜택": best_match["혜택"],
            "유사도 점수": round(float(best_score), 4)
        }
    return None


def get_infra_info(region, infra_df):
    infra_info = infra_df[infra_df["지역"] == region]
    # 🔹 해당 지역의 데이터 필터링
    if not infra_info.empty:
        return infra_info  # ✅ DataFrame 그대로 반환
    else:
        return None  # 데이터가 없으면 None 반환

def get_eco_info(region, eco_df):
    eco_info = eco_df[eco_df["지역"] == region]
    if not eco_info.empty:
        return eco_info
    else:
        return None
# 기업 데이터 기반 추천 시스템
def recommend_region_for_business(business_df, industry_df,regulatory_df,opportunity_df,local_develop_df,infra_df,eco_df):
    specialized_street_data = pd.read_csv("data/specialized_street_data.csv", encoding="utf-8")
    specialized_street_data["대지역"] = specialized_street_data["소재지도로명"].apply(extract_main_region)
    print(specialized_street_data)
    recommendations = []
    for _, row in business_df.iterrows():
        specialize = row.get("특구종류", None)
        print(specialize)
        industry = row["산업군"]
        tax1= row["법인세"]
        tax3= row["재산세"]
        land = row["부지크기"]
        bill = row["임대료"]*10000000
        print(industry)
        business_desc = row["사업 설명"]
        print(business_desc)
        investment = row["투자 규모 (억원)"]
        infra1 = None
        infra2 = None
        infra3 = None
        if len(row["인프라"])> 0 :
            infra1 = row["인프라"][0]
        elif len(row["인프라"])> 1:
            infra2 = row["인프라"][1]
        elif len(row["인프라"])> 2:
            infra3 = row["인프라"][2]
        business_size = classify_business_size(investment)

        if specialize == "규제자유특구":
            regulatory_zones = find_best_matching_regulatory_zone(business_desc, regulatory_df)
            if regulatory_zones:
                # 유사도 점수를 기준으로 내림차순 정렬 후 상위 3개 선택
                top_zones = sorted(regulatory_zones, key=lambda zone: zone["유사도 점수"], reverse=True)[:3]
                candidates = []  # 각 후보 정보를 저장할 리스트
                for zone in top_zones:
                    region = zone["지역명"]
                    infra_data = get_infra_info(region, infra_df)# 해당 지역의 인프라 정보 가져오기
                    eco_data = get_eco_info(region,eco_df)
                    if eco_data is not None:
                        if isinstance(eco_data, pd.DataFrame):
                            eco_data = eco_data.iloc[0]
                        eco_data["평당임대료"] = eco_data["평당임대료"].astype(float)
                        eco_data["부지혜택"] = eco_data["부지혜택"].astype(float)
                        land = float(land)
                        sailbill = eco_data["부지혜택"]*eco_data["평당임대료"]*land
                        current_bill = sailbill/ bill
                        sailtax1 = tax1*eco_data["법인세혜택"]
                        sailtax3 = tax3*eco_data["재산세혜택"]
                        taxresult = tax1+tax3
                        sailtaxresult = sailtax1+sailtax3
                        ssresult = taxresult -sailtaxresult
                        percentagetax = (taxresult - sailtaxresult)/taxresult
                    if infra_data is not None:
                        # 만약 infra_data가 DataFrame이면, 첫 번째 행만 사용하여 Series로 변환
                        if isinstance(infra_data, pd.DataFrame):
                            infra_data = infra_data.iloc[0]
                        # 주거점수 계산
                        livingscore = (
                                (infra_data["아파트 현황"] * 0.1
                                 - infra_data["30년이상 노후주택 분포현황"] * 0.01
                                 + infra_data["연립 및 다세대 주택 현황"])
                                * infra_data["주택당 평균 가구원"]
                        )
                        # 교통점수 계산
                        busscore = infra_data["1인당 자동차 등록대수"] * (
                                infra_data["전기차충전소"] * 0.1 +
                                infra_data["통근통학 인구변화"] * 0.1
                        )
                        # 문화시설점수 계산
                        infrascore = (
                                infra_data["공공도서관 분포현황"] -
                                infra_data["문화시설 1개당 인구 현황"] * 0.001
                        )
                        if infra1 is not None:
                            livingscore = livingscore*2
                        if infra2 is not None:
                            busscore = busscore*2
                        if infra3 is not None:
                            infrascore = infrascore*2

                        finalscore = (livingscore + busscore + infrascore) / 10000 * zone["유사도 점수"]
                        if zone["지역명"] == "부산":
                            finalscore = finalscore/5
                        elif zone["지역명"] == "대구":
                            finalscore = finalscore/5
                        # 계산된 점수를 스칼라(float) 값으로 변환
                        livingscore = float(livingscore)
                        busscore = float(busscore)
                        infrascore = float(infrascore)
                        finalscore = float(finalscore)
                    else:
                        livingscore = None
                        busscore = None
                        infrascore = None
                        finalscore = None
                    sailbill = int(sailbill)
                    candidate = {
                        "기업명": row["기업명"],
                        "산업군": industry,
                        "추천 지역": region,
                        "추천 거리": "규제특구 추천에 따른 거리 추천 없음",
                        "추천 규제특구": zone,  # 후보 특구 정보
                        "투자 규모": investment,
                        "유사도 점수": zone["유사도 점수"],
                        "주거점수": livingscore,
                        "교통점수": busscore,
                        "문화시설점수": infrascore,
                        "최종점수": finalscore,
                        "원래세금": taxresult,
                        "절약세금": sailtaxresult,
                        "내는세금" : ssresult,
                        "percentagetax": percentagetax,
                        "현재부지비용":bill,
                        "할인부지비용":sailbill,
                        "percentagebill":current_bill,
                    }
                    candidates.append(candidate)

                # 최종점수가 None이면 매우 낮은 값(-무한대)으로 처리하여 정렬
                sorted_candidates = sorted(
                    candidates,
                    key=lambda c: c["최종점수"] if c["최종점수"] is not None else float('-inf'),
                    reverse=True
                )
                # 정렬된 모든 후보 정보를 recommendations에 추가
                recommendations.extend(sorted_candidates)
            break
        elif specialize == "기획발전특구":
            opportunity_zone = find_best_matching_opportunity_zone(business_desc, opportunity_df)
            if opportunity_zone:
                recommendations.append({
                    "기업명": row["기업명"],
                    "산업군": industry,
                    "추천 지역": opportunity_zone["기회발전특구사업"],
                    "추천 거리": "기회발전특구 추천에 따른 거리 추천 없음",
                    "추천 기회발전특구": opportunity_zone,
                    "투자 규모": investment
                })
                break
        elif specialize == "지역특화발전특구":
            local_develop_zone = find_best_matching_develop_zone(business_desc, local_develop_df)
            if local_develop_zone:
                recommendations.append({
                    "기업명": row["기업명"],
                    "산업군": industry,
                    "추천 지역": local_develop_zone["지역개발특구"],
                    "x":local_develop_zone["x"],
                    "y":local_develop_zone["y"],
                    "혜택":local_develop_zone["혜택"],
                    "유사도점수":local_develop_zone["유사도 점수"],
                })
                break
        elif specialize == None:
            relevant_industries = get_top_industries_by_business_size(industry_df, business_size)
            print(get_top_industries_by_business_size(industry_df, business_size))
            matching_region = relevant_industries[
                relevant_industries["산업별(10차)대분류"].astype(str).str.contains(industry, na=False, case=False)]
            print(matching_region)
            if not matching_region.empty:
                recommended_region = matching_region.sample(1)["지역별"].values[0]
            else:
                recommended_region = "추천 지역 없음"
            # 산업군 + 사업 설명을 반영한 거리 추천
            recommended_street = recommend_specialized_street(recommended_region, industry, business_desc,
                                                              specialized_street_data)

            recommendations.append({
                "기업명": row["기업명"],
                "산업군": industry,
                "추천 지역": recommended_region,
                "추천 거리": recommended_street,
                "투자 규모": investment
            })
    print(pd.DataFrame(recommendations))
    return pd.DataFrame(recommendations)