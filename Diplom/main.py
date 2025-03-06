# Импорт необходимых модулей
from typing import List, Optional, Literal
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime,
    ForeignKey, Boolean, Table, select, delete
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession
)
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Annotated

# Базовая конфигурация
Base = declarative_base()
app = FastAPI()

# Настройки CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройки базы данных
engine = create_async_engine("sqlite+aiosqlite:///Diplom.db")
new_session = async_sessionmaker(engine, expire_on_commit=False)

# Настройки аутентификации
SECRET_KEY = "5"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Модели базы данных
user_pet_association = Table(
    "user_pet_association",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("pet_id", Integer, ForeignKey("pets.id"), primary_key=True),
)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), index=True)
    email = Column(String(100), unique=True, index=True)
    password = Column(String(255))
    pets = relationship("Pet", secondary=user_pet_association, back_populates="users")


class Pet(Base):
    __tablename__ = "pets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), index=True)
    type = Column(String(50))
    breed = Column(String(50), nullable=True)
    birth_date = Column(Date, nullable=True)
    gender = Column(String(10))
    age = Column(Integer)
    users = relationship("User", secondary=user_pet_association, back_populates="pets")
    health_records = relationship("HealthRecord", back_populates="pet", cascade="all, delete")
    vaccinations = relationship("Vaccination", back_populates="pet", cascade="all, delete")
    reminders = relationship("Reminder", back_populates="pet", cascade="all, delete")
    feeding_schedules = relationship("FeedingSchedule", back_populates="pet", cascade="all, delete")


class HealthRecord(Base):
    __tablename__ = "health_records"
    id = Column(Integer, primary_key=True, index=True)
    pet_id = Column(Integer, ForeignKey("pets.id"))
    record_date = Column(Date, default=date.today)
    weight = Column(Float)
    description = Column(String(500))
    pet = relationship("Pet", back_populates="health_records")


class Vaccination(Base):
    __tablename__ = "vaccinations"
    id = Column(Integer, primary_key=True, index=True)
    pet_id = Column(Integer, ForeignKey("pets.id"))
    name = Column(String(100))
    date_administered = Column(Date)
    next_date = Column(Date, nullable=True)
    repeated = Column(Boolean, default=False)
    pet = relationship("Pet", back_populates="vaccinations")


class Reminder(Base):
    __tablename__ = "reminders"
    id = Column(Integer, primary_key=True, index=True)
    pet_id = Column(Integer, ForeignKey("pets.id"))
    description = Column(String(200))
    reminder_date = Column(DateTime)
    is_completed = Column(Boolean, default=False)
    pet = relationship("Pet", back_populates="reminders")


class FeedingSchedule(Base):
    __tablename__ = "feeding_schedules"
    id = Column(Integer, primary_key=True, index=True)
    pet_id = Column(Integer, ForeignKey("pets.id"))
    feeding_time = Column(DateTime)
    food_type = Column(String(100))
    quantity = Column(String(50))
    notes = Column(String(200), nullable=True)
    pet = relationship("Pet", back_populates="feeding_schedules")


# Pydantic модели
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class UserBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., max_length=100)
    password: str = Field(..., min_length=6, max_length=255)


class UserResponse(BaseModel):
    id: int
    name: str
    email: str

    class Config:
        from_attributes = True


class PetBase(BaseModel):
    name: str = Field(..., max_length=50)
    type: str = Field(..., max_length=50)
    breed: Optional[str] = Field(None, max_length=50)
    birth_date: Optional[date] = None
    gender: Literal["male", "female", "other"]
    age: int = Field(..., gt=0)


class PetResponse(BaseModel):
    id: int
    name: str
    type: str
    breed: Optional[str]
    birth_date: Optional[date]
    gender: str
    age: int

    class Config:
        from_attributes = True


class ReminderBase(BaseModel):
    description: str = Field(..., max_length=200)
    reminder_date: datetime
    is_completed: bool = False


class ReminderResponse(ReminderBase):
    id: int
    pet_id: int

    class Config:
        from_attributes = True


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=6, max_length=255)


class ReminderCreate(BaseModel):
    pet_id: int
    description: str = Field(..., max_length=200)
    reminder_date: datetime
    is_completed: bool = False


# Зависимости
async def get_session() -> AsyncSession:
    async with new_session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


# Функции
def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def authenticate_user(email: str, password: str, session: AsyncSession) -> Optional[User]:
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar()
    if not user or not pwd_context.verify(password, user.password):
        return None
    return user


async def get_current_user(
        token: Annotated[str, Depends(oauth2_scheme)],
        session: SessionDep
) -> Optional[User]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный токен",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar()
    if user is None:
        raise credentials_exception
    return user


# Эндпоинты
@app.post("/rebuild_bd")
async def rebuild_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    return {"status": "Database rebuilt"}


@app.get("/check-token")
async def check_token(current_user: Annotated[User, Depends(get_current_user)]):
    return {"status": "valid"}


@app.post("/register", response_model=UserResponse)
async def register_user(user_data: UserBase, session: SessionDep):
    existing_email = await session.execute(select(User).where(User.email == user_data.email))
    if existing_email.scalar():
        raise HTTPException(status_code=400, detail="Email уже зарегистрирован")

    hashed_password = pwd_context.hash(user_data.password)
    new_user = User(name=user_data.name, email=user_data.email, password=hashed_password)
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user


@app.post("/token", response_model=Token)
async def login_for_access_token(
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
        session: SessionDep
):
    user = await authenticate_user(form_data.username, form_data.password, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user

@app.patch("/users/me", response_model=UserResponse)
async def update_user_profile(
    update_data: UserBase,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_user)]
):
    try:
        current_user.name = update_data.name
        current_user.email = update_data.email
        await session.commit()
        await session.refresh(current_user)
        return current_user
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pets", response_model=PetResponse)
async def create_pet(
        pet: PetBase,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Неверный токен")

    try:
        # Создаем нового питомца
        new_pet = Pet(**pet.model_dump())
        session.add(new_pet)
        await session.flush()  # Получаем ID нового питомца

        # Явно добавляем связь в ассоциативную таблицу
        stmt = user_pet_association.insert().values(
            user_id=current_user.id,
            pet_id=new_pet.id
        )
        await session.execute(stmt)

        await session.commit()
        await session.refresh(new_pet)
        return new_pet
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=422,
            detail=f"Ошибка создания питомца: {str(e)}"
        )


@app.delete("/pets/{pet_id}")
async def delete_pet(
        pet_id: int,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    # Проверяем принадлежность питомца пользователю
    result = await session.execute(
        select(user_pet_association)
        .where(user_pet_association.c.user_id == current_user.id)
        .where(user_pet_association.c.pet_id == pet_id)
    )
    if not result.scalar():
        raise HTTPException(status_code=403, detail="Нет прав на удаление")

    try:
        # Удаляем связи в ассоциативной таблице
        await session.execute(
            delete(user_pet_association)
            .where(user_pet_association.c.pet_id == pet_id)
        )

        # Удаляем самого питомца (каскадное удаление сработает автоматически)
        await session.execute(
            delete(Pet)
            .where(Pet.id == pet_id)
        )

        await session.commit()
        return {"status": "Питомец удален"}

    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Ошибка удаления: {str(e)}")


@app.get("/pets", response_model=List[PetResponse])
async def get_user_pets(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_user)]
):
    result = await session.execute(
        select(Pet)
        .join(user_pet_association, Pet.id == user_pet_association.c.pet_id)
        .where(user_pet_association.c.user_id == current_user.id)
    )
    return result.scalars().all()


@app.post("/pets/{pet_id}/users/{user_id}")
async def add_user_to_pet(
        pet_id: int,
        user_id: int,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    pet = await session.get(Pet, pet_id)
    if not pet:
        raise HTTPException(status_code=404, detail="Питомец не найден")

    result = await session.execute(
        select(user_pet_association)
        .where(user_pet_association.c.user_id == current_user.id)
        .where(user_pet_association.c.pet_id == pet_id)
    )
    if not result.scalar():
        raise HTTPException(status_code=403, detail="Нет доступа к питомцу")

    user_to_add = await session.get(User, user_id)
    if not user_to_add:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    result = await session.execute(
        select(user_pet_association)
        .where(user_pet_association.c.user_id == user_id)
        .where(user_pet_association.c.pet_id == pet_id)
    )
    if result.scalar():
        raise HTTPException(status_code=400, detail="Пользователь уже добавлен")

    await session.execute(
        user_pet_association.insert().values(user_id=user_id, pet_id=pet_id)
    )
    await session.commit()
    return {"status": "Пользователь успешно добавлен"}


@app.delete("/pets/{pet_id}/users/{user_id}")
async def remove_user_from_pet(
        pet_id: int,
        user_id: int,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    pet = await session.get(Pet, pet_id)
    if not pet:
        raise HTTPException(status_code=404, detail="Питомец не найден")

    result = await session.execute(
        select(user_pet_association)
        .where(user_pet_association.c.user_id == current_user.id)
        .where(user_pet_association.c.pet_id == pet_id)
    )
    if not result.scalar():
        raise HTTPException(status_code=403, detail="Нет доступа к питомцу")

    result = await session.execute(
        select(user_pet_association)
        .where(user_pet_association.c.user_id == user_id)
        .where(user_pet_association.c.pet_id == pet_id)
    )
    if not result.scalar():
        raise HTTPException(status_code=404, detail="Связь не найдена")

    await session.execute(
        delete(user_pet_association)
        .where(user_pet_association.c.user_id == user_id)
        .where(user_pet_association.c.pet_id == pet_id)
    )
    await session.commit()
    return {"status": "Пользователь успешно удален"}


@app.get("/reminders", response_model=List[ReminderResponse])
async def get_reminders(
        start: date,
        end: date,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    result = await session.execute(
        select(Reminder)
        .join(Pet, Reminder.pet_id == Pet.id)
        .join(user_pet_association, Pet.id == user_pet_association.c.pet_id)
        .where(user_pet_association.c.user_id == current_user.id)
        .where(Reminder.reminder_date >= start)
        .where(Reminder.reminder_date <= end)
    )
    return result.scalars().all()


@app.post("/reminders", response_model=ReminderResponse)
async def create_reminder(
        reminder: ReminderCreate,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    result = await session.execute(
        select(user_pet_association)
        .where(user_pet_association.c.user_id == current_user.id)
        .where(user_pet_association.c.pet_id == reminder.pet_id)
    )
    if not result.scalar():
        raise HTTPException(status_code=403, detail="Нет доступа к питомцу")

    new_reminder = Reminder(**reminder.model_dump())
    session.add(new_reminder)
    await session.commit()
    await session.refresh(new_reminder)
    return new_reminder


@app.post("/users/change-password")
async def change_password(
        change_data: ChangePasswordRequest,
        current_user: Annotated[User, Depends(get_current_user)],
        session: SessionDep
):
    if not pwd_context.verify(change_data.old_password, current_user.password):
        raise HTTPException(status_code=400, detail="Неверный текущий пароль")

    current_user.password = pwd_context.hash(change_data.new_password)
    await session.commit()
    return {"status": "Пароль успешно изменен"}


# Запуск приложения
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
