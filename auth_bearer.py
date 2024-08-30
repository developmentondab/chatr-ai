import os
from datetime import datetime
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from jose import jwt
import mysql_db

ALGORITHM = "HS256"
JWT_SECRET_KEY = os.environ['JWT_SECRET_KEY']

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        if credentials:
            # print(credentials)
            if not credentials.scheme == "Bearer":
                raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(status_code=403, detail="Invalid token or expired token.")
            self.check_db_token(credentials.credentials)
            return self.get_user_data(credentials.credentials)
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")

    def verify_jwt(self, jwtoken: str) -> bool:
        isTokenValid: bool = False

        try:
            payload = jwt.decode(jwtoken, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        except:
            payload = None
        if payload:
            isTokenValid = True
        return isTokenValid

    def get_user_data(self, jwtoken: str):
        try:
            payload = jwt.decode(jwtoken, JWT_SECRET_KEY, algorithms=[ALGORITHM])

            if datetime.fromtimestamp(payload['exp']) < datetime.now():
                raise HTTPException(status_code=401, detail="Token expired")

            return payload['sub_id']
        except:
            raise HTTPException(status_code=401, detail="Invalid authorization code.")
            
    def check_db_token(self, jwtoken: str):
        token_res = mysql_db.check_token(jwtoken)     
        if token_res:
            raise HTTPException(status_code=401, detail="Invalid authorization code.")
        
        return True
