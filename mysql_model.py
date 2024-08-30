from pydantic import BaseModel

class user(BaseModel):
    first_name: str
    last_name: str = None
    username: str
    password: str
    email: str

class collection(BaseModel):
    user_id: int
    collection: str
    kb_name: str = None
    data_type: str
