# 서빙
![project_architecture](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-12/blob/main/assets/image/project_architecture.png)

## 환경
||version|
|---|---|
|Python|3.9.0|
|npm|6.14.18|

상세 버전은 `pyproject.toml`(Backend) 및 `server/frontend/package.json`(Frontend) 참조

## 구조
```
server/
├── README.md
├── backend/
|   ├── config/
|   |   └── settings.py         // .env 로드
|   ├── app.py                  // FastAPI 서버 생성 및 라우팅
|   ├── mary.py                 // 모델 로드 및 프롬프팅
|   ├── models.py               // pydantic BaseModel 정의
|   └── welcome_msg.txt
|
└──frontend/
    ├── package.json
    ├── public/
    |   ├── index.html
    |   └── mary.svg
    └── src/
        ├── App.js              // React 애플리케이션 정의
        ├── App.css             
        ├── index.js
        ├── index.css
        ├── message.js          // 메시지 컴포넌트 생성 스크립트
        └── randomString.js     // 컴포넌트 id 생성 스크립트
```
## 환경변수 설정
### Backend
루트 디렉토리의 `.env.template`를 `.env`로 복사한 후 환경변수를 작성합니다. 작성한 환경변수는 `config/settings.py`에서 로드해 사용됩니다.

|변수명|설명|
|---|---|
|OPENAI_API_KEY|임베딩을 생성하기 위한 OPENAI api key|
|HOST|FastAPI 서버를 위한 호스트 주소(미설정시 `0.0.0.0`)|
|PORT|FastAPI 서버를 위한 포트 번호(미설정시 `8000`)|
|REACT_PORT|React 서버를 위한 포트 번호(미설정시 `3000`)|
|DB_URL|MongoDB Atlas 연결 URI|
|DB_NAME|MongoDB Atlas 데이터베이스명|
|COLLECTION|사용자 메시지 데이터 저장 컬렉션명|
|USERNAME|DB 유저명, `DB_URL`에 포함되어 있는 경우 작성하지 않아도 무방|
|PASSWORD|DB 패스워드, `DB_URL`에 포함되어 있는 경우 작성하지 않아도 무방|
|MODEL_NAME|로드할 모델 이름|
|MODEL_PATH|로드할 모델이 저장되어 있는 경로|
|EMBEDDING_SIZE|임베딩 차원 크기|
|VECTOR_INDEX_COLLECTION|연결된 MongoDB Atlas 데이터베이스에서 Vector Search를 수행할 컬렉션명|

### Frontend
`package.json`의 `proxy`에 FastAPI 서버 주소를 입력한 후 `npm install`을 실행합니다.

## DB
2개의 컬랙션을 필요로 하며, 각 컬렉션의 스키마는 다음과 같습니다. 컬렉션명은 필요에 따라 임의로 설정 가능합니다.

### user_data
메시지 텍스트를 개별 데이터로 저장합니다. 프론트엔드 렌더링 시 조회하여 사용합니다.
|field|type|discription|
|---|---|---|
|_id|ObjectID|MongDB Object ID|
|msg_text|String|개별 메시지 텍스트|
|bot|Boolean|`true`일시 챗봇 메시지, `false`일시 유저 메시지|
|user_id|String|세션 식별용 유저 해시, 16자리|

### message_store
`### 명령어` - `### 응답` 쌍으로 메시지 텍스트와 임베딩 정보를 저장합니다. 유저가 새 메시지를 입력하면 해당 컬렉션에서 새 메시지를 기반으로 Vector Search를 수행해 가장 관련도가 높은 2개의 메시지를 검색해 메시지 생성 시 맥락으로 반영합니다.

Vector Search를 사용하기 위해 사전 Vector Index 설정이 필요합니다.

|field|type|discription|
|---|---|---|
|_id|ObjectID|MongDB Object ID|
|text|String|`### 명령어` - `### 응답` 텍스트 쌍|
|embedding|Array|메시지 텍스트 임베딩 벡터, 1536 차원|
|user_id|String|세션 식별용 유저 해시, 16자리|
|timstamp|Double|메시지 생성 시각 타임스탬프, UNIX 타임스탬프|

## API 리스트
|Method|Path|Description|
|---|:---:|---|
|GET|/|클라이언트에 저장된 쿠키를 바탕으로 현재 세션에 해당하는 모든 대화 메시지를 `user_message`에서 조회 후 반환, 클라이언트에 저장된 세션 정보가 없을 경우 새로운 세션 정보를 설정하고 웰컴 메시지 중 하나를 임의로 반환|
|POST|/|사용자 입력 메시지를 DB에 저장하고 메시지와 세션 정보를 바탕으로 응답 메시지 생성 후 반환|
|DELETE|/|현재 세션에 해당하는 모든 대화 메시지를 DB에서 삭제 후 세션 정보 리셋, GET 리다이렉트|