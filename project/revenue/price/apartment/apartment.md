# 아파트 실거래가 예측

**배경**  

통계청 2015년 자료에 의하면 (https://bit.ly/2SFyzMA)  
일반적인 한국인의 절반은 48.1%는 아파트에 살고 있습니다.  
그들은 아파트 주거 선호도가 매우 높습니다.  
또한 부의 증식 수단으로 생각 하기 때문에 아파트 가격에 관심이 많습니다.  

이번 대회의 데이터 제공자는 직방입니다.  
직방은 부동산 정보의 비대칭성과 불투명성을 해소하기 위해 노력하며,  
중개사와 구매자를 연결하여 부동산정보 서비스 시장의 신뢰도를 높이는데 기여합니다.  

최근 매물 가격 정보는 직방, 다음부동산, 네이버부동산에서 볼 수 있습니다.  
하지만 최근 매물 가격은 아직 거래되지 않아 정확하지 않은 정보일 수 있습니다.

이에따라, 본 대회는 실 거래가와 아파트, 학교, 지하철역 정보를 제공하며,  
아파트 구매자들의 비대칭성 정보를 해결하기 위해 미래의 실 거래가 예측을 목표로 합니다.
</br></br>

**목적** 

서울/부산 지역 아파트 실 거래가를 예측하는 모델 개발
</br></br>

**데이터셋**
- 데이터셋 정보   
약1,600,000여개의 실거래 데이터, 아파트 거래일, 지역, 전용면적, 공급면적 등의 정보가 제공됩니다.  
* 국토교통부 실거래가 공개시스템 (http://rt.molit.go.kr/)과 같은 법적인 제약이 없는 외부 데이터(공공 데이터) 사용이 가능합니다.  

- 데이터 파일
1. train.csv/test.csv : 서울/부산 지역의 1,100,000여개 거래 데이터, 아파트 거래일, 지역, 전용면적, 실 거래가 등의 정보 / 실 거래가를 제외하고 train.csv와 동일
    - apartment_d - 아파트 아이디
    - city - 도시
    - dong - 동
    - jibun - 지번
    - apt - 아파트 단지 이름
    - addr_kr - 주소
    - exclusive_use_area - 전용면적
    - year_of_completion - 설립일자
    - transaction_year_month - 거래년월
    - transaction_date - 거래날짜
    - floor - 층
    - transaction_real_price - 실거래가(train만 존재)

2. park.csv : 서울/부산 지역의 공원에 대한 정보
    - city - 도시
    - gu - 구
    - dong - 동
    - park_name - 공원이름
    - park_type - 공원 종류
    - park_area - 공원의 넓이
    - park_exercise_facility - 공원보유 운동시설
    - park_entertainment_facility - 공원보유 유희시설
    - park_benefit_facility - 공원보유 편익시설
    - park_cultural_tacitiy - 공원보유 교양시설
    - park facility-other - 공원보유 기타시설
    - park_open_year - 공원 개장년도
    - reference_date - 데이터 기준일자(해당 데이터가 기록된 일자)

3. day_care_center.csv : 서울/부산 지역의 어린이집에 대한 정보
    - city - 도시
    - gu - 구
    - day_care_name - 어린이집 이름
    - day-care_type - 어린이집 종류
    - day_care_baby_num - 정원수
    - teacher_num - 보육교직원수
    - nursing_room_num - 보육실수
    - playground_num - 놀이터수
    - CCTV_num - CCTV 설치수
    - is_commuting-vehicle - 통학차량 운영여부
    - reference_date - 데이터 기준일자(해당 데이터가 기록된 일자)
