from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
import os
import hashlib
from passlib.context import CryptContext
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.environ.get('SECRET_KEY')
ALGORITHM = os.environ.get('ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = 60
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set")

pwd_context=CryptContext(schemes=["bcrypt"], deprecated="auto")
def _normalize_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def hash_password(password: str) -> str:
    normalized = _normalize_password(password)
    return pwd_context.hash(normalized)

def verify_password(password: str, hashed: str) -> bool:
    normalized = _normalize_password(password)
    return pwd_context.verify(normalized, hashed)

def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None
) -> str:
    if "user_id" not in data:
        raise ValueError("Token payload must include user_id")

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        return payload

    except JWTError as e:
        msg = str(e).lower()

        if "expired" in msg:
            raise HTTPException(status_code=401, detail="Token expired")

        raise HTTPException(status_code=401, detail="Invalid token")

