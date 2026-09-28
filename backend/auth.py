"""Authentication helpers for password and Google sign-in."""

import hashlib
import hmac
import os
import secrets
import time
from typing import Any
from uuid import UUID

import httpx
import jwt
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import User


def _secret() -> str:
    value = os.getenv("AUTH_SECRET_KEY")
    if not value:
        raise RuntimeError("AUTH_SECRET_KEY is required for authentication.")
    return value


def hash_password(password: str) -> str:
    """Hash a password using a salted scrypt digest."""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str | None) -> bool:
    """Verify a password against a stored scrypt digest."""
    if not encoded or not encoded.startswith("scrypt$"):
        return False
    _, salt_hex, digest_hex = encoded.split("$", 2)
    digest = hashlib.scrypt(
        password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def issue_token(user: User) -> str:
    """Issue a short-lived bearer token for a user."""
    return jwt.encode(
        {"sub": str(user.id), "email": user.email, "exp": int(time.time()) + 604800},
        _secret(),
        algorithm="HS256",
    )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a bearer token."""
    try:
        return jwt.decode(token, _secret(), algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


def user_from_token(session: Session, token: str) -> User:
    """Resolve a bearer token to its database user."""
    claims = decode_token(token)
    try:
        user_id = UUID(str(claims["sub"]))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token subject") from exc
    user = session.scalar(select(User).where(User.id == user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def google_profile(id_token: str) -> tuple[str, str]:
    """Validate a Google identity token and return its subject and email."""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            "https://oauth2.googleapis.com/tokeninfo", params={"id_token": id_token}
        )
    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    data = response.json()
    email = str(data.get("email", "")).strip().lower()
    subject = str(data.get("sub", "")).strip()
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not email or not subject or (client_id and data.get("aud") != client_id):
        raise HTTPException(status_code=401, detail="Google account validation failed")
    return subject, email
