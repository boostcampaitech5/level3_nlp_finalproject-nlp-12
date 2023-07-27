from pydantic import BaseModel
from typing import Optional

class MsgModel(BaseModel):
    user_id: str
    msg_text: str
    bot: bool

    class Config:
        schema_extra = {
            "example": {
                "user_id": "test",
                "msg_text": "hello, world!",
                "bot": True,
            }
        }

class MsgHistoryModel(BaseModel):
    msg_list: list
    user_id: str