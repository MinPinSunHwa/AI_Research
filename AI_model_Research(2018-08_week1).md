# Objective  

앞으로 2시간 이후의 암호화폐 가격 예측을 위해서, 2시간 단위의 거래가격, 거래량 정보를 기본으로 학습을 시켜서 모델링하고,   
그 외의 변수들을 다양하게 학습데이터에 반영하여 총 10가지의 모델들이 가격예측정확도를 경쟁하도록 만든다.


--------------------------


# Our Test model
##### WSAEs-LSTM = Wavelet Transform + Stacked Autoencoder + LSTM From
##### Bao W, Yue J, Rao Y (2017) A deep learning framework for financial time series using stacked autoencoders and long-short term memory. PLOS ONE 12(7): e0180944.
![alt text](./Images/W_Bao_et_al.PNG)

### 참조 모델
##### Ciaian P, Rajcaniova M, Kancs d'A(2016) The Economics of BitCoin Price Formation Applied Economics 48(19):1799-1815
> 1 Supply-Demend Fundamentals  
2 Wikipedia View  
3 Global financial indicators


------------------------------------------------------------------------
# GPU Computing
## GPU Framework
### 1. Google Tensorflow  
가장 대중적인 프레임워크, 시중에 많은 교재들이 있고, Stackoverflow 등에 실제 사용 사례들도 많이 올라와 있는 장점이 있다. 다만 계산 그래프가 Python으로 만들어져 있어서 다른 프레임워크보다 속도가 느린 단점이 있다. 
### 2. Apache MXNet  
아파치 재단의 오픈소스 프레임워크. 아마존의 CTO가 직접 대놓고 소개할 정도로 AWS에서 밀어주는 프레임워크로 좋은 성능을 보이는 것으로 알려짐  
### 3. Gluon  
작년 10월에 AWS와 Microsoft가 공동 개발, 발표한 High-level 인터페이스로 Keras보다 더 직관적이고 쉬운 것으로 알려져 있다.  
Keras (based on Tensorflow) vs Gluon (based on MXNet)  
=> Google vs Amazon + Microsoft로 대결구도가 잡히는 것 같음  
## GPU Cloud Service  
### 1. Amazon Web Service  
#### 1-1 기본 P3 인스턴스  
최신 Nvidia Tesla Volta GPU를 1, 4, 8대 선택할 수 있으며, 직접 Setting을 해야 한다.  
#### 1-2 Amazon SageMaker  
머신러닝에 특화된 서비스, 하드웨어 사양은 기본 P3와 똑같지만, 각종 딥러닝 프레임워크의 설치 및 성능 최적화까지 완료되어 있다.
#### 1-3 최대 가동시 요금  
| Instance                  | GPU | GPU memory | 1 month (USD) | 1 yr (USD) |
| ------------------------- | --- | ---------- | ------------- | ---------- |
| p3.2xlarge                | 1   |   16 GB    |  3646.1       | 43753.2    |
| ml.p3.2xlarge (SageMaker) | 1   |   16 GB    |  5104.236     | 61250.832  |

### 2. Google CLOUD
저렴한 가격과 자원 사용 현황을 파악하기 편한 깔끔한 웹 인터페이스를 제공한다는 것이 장점. Nvidia Tesla Volta 급의 최신 GPU를 제공해주는 location이 미중부와 타이완에 있고, 후발 주자라는 점에서 안정성이 의심되고 MXNet을 지원해주는지 여부가 불투명    

| Instance                     | GPU | GPU memory | 1 month (USD) | 1 yr (USD) |
| ---------------------------- | --- | ---------- | ------------- | ---------- |
| custom-8-62-extended (Taiwan) | 1   |   16 GB    |  557.63      | 6331.5     |  
### 3. MS Azure  
Google Cloud의 단점을 그대로 가지고 있으며, 딥러닝 서비스의 안정성이 검증되지 않았음

### 4. Naver Cloud
Nvidia Tesla Pascal GPU만 제공해주기 때문에, 성능이 다른 클라우드보다 떨어질 수 밖에 없음   

---------------------------------------------------


# 데이터 수집
##### 종류 : OHLC(Open High Low Close) price, Trading Volume, Market issues, etc
OHLC의 경우는 JSON, csv와 같은 준정형데이터(Semi-structured data)형태로 가지고 올 수 있어서 쉽게 정형데이터(Structured data)로 가공할 수 있음

#### Source 1 : xe.com  
실시간 Exchange rate 뿐만 아니라 Historic hourly data도 제공, REST API만 날리면 json혹은 csv방식으로 data response를 받을 수 있다. 단점은 암호화폐는 Bitcoin만 가능하고, 원하는 시간의 중간 가격을 제공해서 OHLC를 정확히 구할 수 없음, 거래량을 제공해 주지 않음

> curl -i -u account_id:api_key "https://xecdapi.xe.com/v1/convert_from.json/?from=XBT&to=USD&amount=1"

#### Source 2 : [Kaggle-Bitcoin Historical Data](https://www.kaggle.com/mczielinski/bitcoin-historical-data)  
각종 dataset을 제공해주는 Kaggle에서 Bitcoin Historycal data csv파일을 제공해준다.
2012년 1월 부터 현재까지 OHLCV를 제공, 단점은 공신력에 의문이 있고, 1분 단위 데이터라는 점


JSON 형식
> {'price_close': 6.31,
  'price_high': 7.1,
  'price_low': 5.52,
  'price_open': 6.2,
  'time_close': '2012-01-25T21:37:14.0000000Z',
  'volume_traded': 993.32690311}, ... 
  
#### 테이블 스키마(schema)
##### OHLC_BTCvsUSD (2hrs)

| Close       | High       | Low       | Open       | Date       | Volume        |
| ----------- | ---------- | --------- | ---------- | ---------- | ------------- |
| 6.31        | 7.1        | 5.52      | 6.2        | 2012-01-25 | 993.32690311  |
| ....        |....        | ....      | ...        | ....       | ....          |         

##### Market_issues (From http://www.decenter.kr/ etc)

| Date        | Issues                                                                             |
| ----------- | ---------------------------------------------------------------------------------- |
| 2018-07-19  | '美 청문회, 긍정적 목소리 높아…"암호화폐는 사회적 산물, 미래 인프라로 지원해야" ' |
| 2018-07-20  | '美 금융소비자보호국, 블록체인 규제 샌드박스 시행'                                 |
| ....        | ....                                                                               |

##### Hash_rate  (From https://blockchain.info/q/hashrate)

| Date        |  Hash_rate   |
| ----------- | ------------ |
| 2018-03-29  | 26162835     | 
| 2018-03-30  | 27884074     |
| ....        | ....         |

##### Macroeconomic indicators (From quandl.com)

| Date        | US_Crude_Oil   |  US_Gov_Interest_rate | US_Treasury_yield_10yr | ... |
| ----------- | -------------- | --------------------- | ---------------------- | --- |
| 2012-01-20  |   98.15        |   0.05                |  2.05                  | ... |
| ....        | .....          | ......                | ......                 | ... |

##### Volume of daily BitCoin views on Wikipedia (From [Wiki PageViews Analysis](https://tools.wmflabs.org/pageviews/?project=en.wikipedia.org&platform=all-access&agent=user&range=latest-20&pages=Cat|Dog) )

| Date        |  Views      |
| ----------- | ----------- |
| 2015-07-01  | 12957       |
| 2015-07-02  | 9802        |
| ....        | ....        |

MongoDB의 경우 데이터를 BSON으로 저장하고, 스키마를 사전에 정의하지 않아도 되지만 
테이블 JOIN이 되지 않고 Embedded document라는 별도의 방식으로 지원하므로 연구가 필요
예를 들어 위의 Date 컬럼을 이용해서 JOIN을 할 수 없다.
