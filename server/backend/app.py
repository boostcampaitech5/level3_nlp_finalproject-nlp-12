from fastapi import FastAPI, Cookie, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn

from config.settings import settings
from models import MsgModel, MsgHistoryModel

from typing import Optional
import secrets


app = FastAPI()

# React 서버 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins       = [f'http://localhost:{settings.REACT_PORT}'],
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
    app.mongodb_client = AsyncIOMotorClient(
        settings.DB_URL,
        username=settings.USERNAME,
        password=settings.PASSWORD
    )
    app.mongodb = app.mongodb_client[settings.DB_NAME][settings.COLLECTION]

    # TO-DO: 모델 로드 및 초기화 코드


@app.on_event('shutdown')
async def shutdown():
    '''
    FastAPI 서버 종료
    '''

    # MongoDB 클라이언트 연결 종료
    app.mongodb_client.close()


@app.get('/', response_model=MsgHistoryModel)
async def load_conversation(response: Response, user_id: Optional[str] = Cookie(None)):
    '''
    유저 및 메시지 히스토리 조회
    '''
    if not user_id:
        user_id = secrets.token_hex(16)
        response.set_cookie(key='user_id', value=user_id, httponly=True)
        await init_user(user_id)

    # user_id에 해당하는 전체 메시지 히스토리를 DB에서 조회
    messages = []
    for doc in await app.mongodb.find({'user_id':user_id}).to_list(length=300):
        messages.append(to_dict(doc))
    
    history = MsgHistoryModel(msg_list=messages, user_id=user_id)
    return history


@app.post('/', response_description='add_new_message', response_model=MsgModel)
async def message_input(request: MsgModel) -> JSONResponse:
    '''
    유저 메시지 처리
    '''
    await insert_msg(request)
    inserted_bot_msg = await dummy_response(request.user_id, request.msg_text)
    bot_response = await app.mongodb.find_one({'_id': inserted_bot_msg})
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=to_dict(bot_response))


@app.delete('/')
async def reset_conversation() -> RedirectResponse:
    '''
    TO-DO: 현재 유저 초기화 및 DB에서 해당 유저 메시지 삭제
    '''
    return RedirectResponse(url='/', status_code=200, headers={})


async def insert_msg(msg: MsgModel):
    '''
    DB에 메시지 insert
    '''
    add_msg = await app.mongodb.insert_one(msg.dict())
    return add_msg.inserted_id


async def init_user(user_id: str):
    '''
    cookie에 user_id가 설정되어 있지 않은 경우 최초 메시지 생성
    '''
    welcome_msg = MsgModel(user_id=user_id, msg_text='nice to meet you...', bot=True)
    insert_new_msg = await insert_msg(welcome_msg)
    return insert_new_msg


def to_dict(msg) -> dict:
    return {
        'id'        : str(msg['_id']),
        'user_id'   : str(msg['user_id']),
        'msg_text'  : str(msg['msg_text']),
        'bot'       : bool(msg['bot'])
    }


async def dummy_response(user_id: str, msg_text:str):
    '''
    모델이 연결되지 않은 경우에 자동으로 유저 메시지에 응답
    모델이 연결되고 나면 사용하지 않음
    '''
    dummy_msg = MsgModel(user_id=user_id, msg_text=f'echo:{msg_text}', bot=True)
    insert_new_msg = await insert_msg(dummy_msg)
    return insert_new_msg


if __name__=='__main__':
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT
    )