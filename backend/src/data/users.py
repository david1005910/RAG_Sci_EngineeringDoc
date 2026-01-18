"""Persistent user storage for authentication with JSON file backend"""

from typing import Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import json
import os
import logging

from src.core.security import get_password_hash, verify_password

logger = logging.getLogger(__name__)

# Data directory for user storage
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


@dataclass
class User:
    """User model"""
    id: str
    email: str
    name: str
    hashed_password: str
    research_field: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_active: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "hashed_password": self.hashed_password,
            "research_field": self.research_field,
            "created_at": self.created_at if isinstance(self.created_at, str) else self.created_at.isoformat(),
            "is_active": self.is_active
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Create User from dictionary"""
        return cls(
            id=data["id"],
            email=data["email"],
            name=data["name"],
            hashed_password=data["hashed_password"],
            research_field=data.get("research_field"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            is_active=data.get("is_active", True)
        )


class UserStore:
    """Persistent user storage with JSON file backend"""

    def __init__(self, users_file: str = USERS_FILE):
        self._users_file = users_file
        self._users: Dict[str, User] = {}
        self._email_to_id: Dict[str, str] = {}
        self._ensure_data_dir()
        self._load_users()

    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self._users_file), exist_ok=True)

    def _load_users(self):
        """Load users from JSON file"""
        if os.path.exists(self._users_file):
            try:
                with open(self._users_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_data in data.get("users", []):
                        user = User.from_dict(user_data)
                        self._users[user.id] = user
                        self._email_to_id[user.email] = user.id
                logger.info(f"Loaded {len(self._users)} users from {self._users_file}")
            except Exception as e:
                logger.error(f"Failed to load users: {e}")
                self._users = {}
                self._email_to_id = {}

    def _save_users(self):
        """Save users to JSON file"""
        try:
            data = {
                "users": [user.to_dict() for user in self._users.values()],
                "updated_at": datetime.utcnow().isoformat()
            }
            with open(self._users_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self._users)} users to {self._users_file}")
        except Exception as e:
            logger.error(f"Failed to save users: {e}")

    def create_user(
        self,
        email: str,
        password: str,
        name: str,
        research_field: Optional[str] = None
    ) -> User:
        """Create a new user and persist to file"""
        if email in self._email_to_id:
            raise ValueError("User with this email already exists")

        user_id = str(uuid.uuid4())
        hashed_password = get_password_hash(password)

        user = User(
            id=user_id,
            email=email,
            name=name,
            hashed_password=hashed_password,
            research_field=research_field
        )

        self._users[user_id] = user
        self._email_to_id[email] = user_id
        self._save_users()  # Persist to file

        logger.info(f"Created new user: {email}")
        return user

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        user_id = self._email_to_id.get(email)
        if user_id:
            return self._users.get(user_id)
        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self._users.get(user_id)

    def verify_user(self, email: str, password: str) -> Optional[User]:
        """Verify user credentials"""
        user = self.get_user_by_email(email)
        if user and verify_password(password, user.hashed_password):
            return user
        return None

    def email_exists(self, email: str) -> bool:
        """Check if email exists"""
        return email in self._email_to_id

    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user information"""
        user = self._users.get(user_id)
        if not user:
            return None

        for key, value in kwargs.items():
            if hasattr(user, key) and key not in ["id", "hashed_password"]:
                setattr(user, key, value)

        self._save_users()
        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        user = self._users.get(user_id)
        if not user:
            return False

        del self._users[user_id]
        del self._email_to_id[user.email]
        self._save_users()
        return True

    def get_all_users(self) -> list:
        """Get all users (for admin purposes)"""
        return list(self._users.values())


# Global user store instance
user_store = UserStore()


def init_default_users():
    """Initialize default test users"""
    default_users = [
        {
            "email": "test@example.com",
            "password": "test1234",
            "name": "Test User",
            "research_field": "암 연구"
        },
        {
            "email": "admin@bio-rag.com",
            "password": "admin1234",
            "name": "Admin",
            "research_field": "면역학"
        },
        {
            "email": "demo@bio-rag.com",
            "password": "demo1234",
            "name": "Demo User",
            "research_field": "유전학"
        }
    ]

    for user_data in default_users:
        try:
            user_store.create_user(
                email=user_data["email"],
                password=user_data["password"],
                name=user_data["name"],
                research_field=user_data["research_field"]
            )
        except ValueError:
            # User already exists
            pass


# Initialize default users on module load
init_default_users()
