from fastapi import FastAPI, Cookie, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import uvicorn

from config.settings import settings
from mary import Mary
from models import MsgModel, MsgHistoryModel

from typing import Optional
import secrets
import random


app = FastAPI()

# React 서버 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins       = [f'http://localhost:{settings.ServerSetting.REACT_PORT}',
                           f'http://{settings.ServerSetting.HOST}:{settings.ServerSetting.REACT_PORT}'],
    allow_credentials   = True,
    allow_methods       = ['GET', 'POST', 'OPTIONS', 'DELETE'],
    allow_headers       = ['*']
)


@app.on_event('startup')
async def startup():
    '''
    FastAPI 서버 실행시 최초로 실행
    - MongoDB 클라이언트 연결
    - 모델 로드 및 초기화
    '''

    # MongoDB 클라이언트 연결
    app.mongodb_client = MongoClient(settings.DatabaseSetting.DB_URL)
    app.mongodb        = app.mongodb_client[settings.DatabaseSetting.DB_NAME]
    
    # 개별 메시지 저장 collection 설정
    app.user_data      = app.mongodb[settings.DatabaseSetting.COLLECTION]
    app.message_store  = app.mongodb[settings.ModelSetting.VECTOR_INDEX_COLLECTION]

    # 모델 로드 및 초기화
    app.mary = Mary(settings.ModelSetting, app.mongodb)

    # 첫번째 메시지 로드
    app.welcome_msg = []
    with open('./welcome_msg.txt', 'r', encoding='utf8') as f:
        app.welcome_msg = f.readlines()


@app.on_event('shutdown')
async def shutdown():
    '''
    FastAPI 서버 종료시 호출
    '''

    # MongoDB 클라이언트 연결 종료
    app.mongodb_client.close()


@app.get('/', response_model=MsgHistoryModel)
def load_conversation(response: Response, user_id: Optional[str] = Cookie(None)):
    '''
    유저 및 메시지 히스토리 조회
    '''
    if not user_id:
        user_id = secrets.token_hex(16)
        response.set_cookie(key='user_id', value=user_id, httponly=True)
        init_user(user_id)

    # user_id에 해당하는 전체 메시지 히스토리를 DB에서 조회
    messages = []
    for doc in app.user_data.find({'user_id':user_id}):
        messages.append(to_dict(doc))
    
    history = MsgHistoryModel(msg_list=messages, user_id=user_id)
    return history


@app.post('/', response_description='add_new_message', response_model=MsgModel)
def message_input(request: MsgModel) -> JSONResponse:
    '''
    유저 메시지 처리
    '''
    insert_msg(request)

    # 유저 메시지에 대한 응답 생성
    bot_response = app.mary.get_response(request.msg_text, request.user_id)

    # 생성된 응답을 DB에 저장
    response_msg = MsgModel(user_id=request.user_id, msg_text=bot_response, bot=True)
    inserted_bot_msg = insert_msg(response_msg)

    # 저장된 응답을 DB에서 다시 불러와 JSONResponse로 반환
    bot_response = app.user_data.find_one({'_id': inserted_bot_msg})
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=to_dict(bot_response))


@app.delete('/')
def reset_conversation(response: Response, user_id: Optional[str] = Cookie(None)) -> Response:
    '''
    TO-DO: 현재 유저 초기화 및 DB에서 해당 유저 메시지 삭제
    '''
    app.user_data.delete_many({'user_id':user_id})
    app.message_store.delete_many({'user_id':user_id})

    response.delete_cookie(key='user_id')
    response.status_code = status.HTTP_200_OK
    return response


def insert_msg(msg: MsgModel):
    '''
    DB에 메시지 insert
    '''
    add_msg = app.user_data.insert_one(msg.dict())
    return add_msg.inserted_id


def init_user(user_id: str):
    '''
    cookie에 user_id가 설정되어 있지 않은 경우 최초 메시지 생성
    '''
    welcome_msg = MsgModel(user_id=user_id,
                           msg_text=random.choice(app.welcome_msg),
                           bot=True)
    insert_new_msg = insert_msg(welcome_msg)
    return insert_new_msg


def to_dict(msg) -> dict:
    return {
        'user_id'   : str(msg['user_id']),
        'msg_text'  : str(msg['msg_text']),
        'bot'       : bool(msg['bot'])
    }


if __name__=='__main__':
    uvicorn.run(
        'app:app',
        host=settings.ServerSetting.HOST,
        port=settings.ServerSetting.PORT,
        timeout_keep_alive=50,
        reload=True
    )