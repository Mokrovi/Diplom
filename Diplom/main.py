from typing import List, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Boolean, Table, select
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from fastapi import FastAPI, Depends, HTTPException, status, Response, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import date, datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


# API
Base = declarative_base()
app = FastAPI()
engine = create_async_engine('sqlite+aiosqlite:///Diplom.db')
new_session = async_sessionmaker(engine, expire_on_commit=False)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# JWT и хэширование
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: str | None = None


def get_password_hash(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


async def authenticate_user(email: str, password: str, session: AsyncSession):
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar()
    if not user or not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Сущности бд
user_pet_association = Table(
    'user_pet_association',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('pet_id', Integer, ForeignKey('pets.id'))
)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    password = Column(String(255))

    pets = relationship("Pet", secondary=user_pet_association, back_populates="users")


class Pet(Base):
    __tablename__ = "pets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), index=True)
    type = Column(String(50))  # dog, cat, etc.
    breed = Column(String(50), nullable=True)
    birth_date = Column(Date, nullable=True)
    gender = Column(String(10))  # male/female/other
    age = Column(Integer)
    user_id = Column(Integer, ForeignKey("users.id"))

    users = relationship("User", secondary=user_pet_association, back_populates="pets")
    health_records = relationship("HealthRecord", back_populates="pet")
    vaccinations = relationship("Vaccination", back_populates="pet")
    reminders = relationship("Reminder", back_populates="pet")
    feeding_schedules = relationship("FeedingSchedule", back_populates="pet")


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
    quantity = Column(String(50))  # e.g., 100g, 1 cup
    notes = Column(String(200), nullable=True)

    pet = relationship("Pet", back_populates="feeding_schedules")


# Schemas
class UserBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., max_length=100)
    password: str = Field(..., min_length=6, max_length=255)


class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True


class PetBase(BaseModel):
    name: str = Field(..., max_length=50)
    type: str = Field(..., max_length=50)
    breed: Optional[str] = Field(None, max_length=50)
    birth_date: Optional[date] = None
    gender: Literal['male', 'female', 'other']
    age: int = Field(..., gt=0)


class PetResponse(PetBase):
    id: int
    user_id: int
    health_records: List[int] = []
    vaccinations: List[int] = []
    reminders: List[int] = []
    feeding_schedules: List[int] = []

    class Config:
        from_attributes = True


class HealthRecordBase(BaseModel):
    record_date: date = Field(default_factory=date.today)
    weight: float = Field(..., gt=0)
    description: str = Field(..., max_length=500)


class HealthRecordResponse(HealthRecordBase):
    id: int
    pet_id: int

    class Config:
        from_attributes = True


class VaccinationBase(BaseModel):
    name: str = Field(..., max_length=100)
    date_administered: date
    next_date: Optional[date] = None
    repeated: bool = False


class VaccinationResponse(VaccinationBase):
    id: int
    pet_id: int

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


class FeedingScheduleBase(BaseModel):
    feeding_time: datetime
    food_type: str = Field(..., max_length=100)
    quantity: str = Field(..., max_length=50)
    notes: Optional[str] = Field(None, max_length=200)


class FeedingScheduleResponse(FeedingScheduleBase):
    id: int
    pet_id: int

    class Config:
        from_attributes = True


async def get_session():
    async with new_session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
PetDep = Annotated[Pet, Depends(lambda pet_id: get_pet(pet_id))]


async def get_pet(pet_id: int, session: SessionDep) -> Pet:  # Поиск питомца
    pet = await session.get(Pet, pet_id)
    if not pet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Питомец не найден"
        )
    return pet


#Разрешения

@app.options("/register")
async def options_register():
    return {"Allow": "POST"}


@app.options("/token")
async def options_token():
    return {"Allow": "POST"}

#Запросы

@app.post("/rebuild_bd")
async def rebuild_bd():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    return {"ok": True}


@app.post("/post_user", response_model=UserResponse)
async def post_user(
    data: UserBase,
    session: SessionDep
):
    new_user = User(
        name=data.name,
        email=data.email,
        password=data.password,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user


# Аутентификация
@app.post("/token", response_model=Token)
async def login_for_access_token(
    session: SessionDep,
    form_data: OAuth2PasswordRequestForm = Depends()
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


# Регистрация
@app.post("/register", response_model=UserResponse)
async def register_user(user_data: UserBase, session: SessionDep):
    existing_email = await session.execute(select(User).where(User.email == user_data.email))
    if existing_email.scalar():
        raise HTTPException(
            status_code=400,
            detail="Email уже зарегистрирован"
        )

    existing_name = await session.execute(select(User).where(User.name == user_data.name))
    if existing_name.scalar():
        raise HTTPException(
            status_code=400,
            detail="Имя пользователя уже занято"
        )
    if len(user_data.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Пароль должен содержать минимум 6 символов"
        )
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        name=user_data.name,
        email=user_data.email,
        password=hashed_password,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user


# Текущий пользователь
async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: SessionDep
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверные учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    result = await session.execute(select(User).where(User.email == token_data.email))
    user = result.scalar()
    if user is None:
        raise credentials_exception
    return user


# Питомцы
@app.post("/pets", response_model=PetResponse)
async def create_pet(
    pet: PetBase,
    session: SessionDep,
    current_user: User = Depends(get_current_user)
):
    new_pet = Pet(**pet.model_dump(), user_id=current_user.id)
    session.add(new_pet)
    await session.commit()
    await session.refresh(new_pet)
    return new_pet


@app.get("/pets", response_model=List[PetResponse])
async def get_all_pets(session: SessionDep):
    result = await session.execute(select(Pet))
    pets = result.scalars().all()
    return pets


# Записи здоровья
@app.post("/health-records", response_model=HealthRecordResponse)
async def create_health_record(
    record: HealthRecordBase,
    session: SessionDep,
    pet: PetDep
):
    new_record = HealthRecord(**record.model_dump())
    session.add(new_record)
    await session.commit()
    await session.refresh(new_record)
    return new_record


@app.get("/pets/{pet_id}/health-records", response_model=List[HealthRecordResponse])
async def get_pet_health_records(pet: PetDep, session: SessionDep):
    return pet.health_records


# Вакцинация
@app.post("/vaccinations", response_model=VaccinationResponse)
async def create_vaccination(
    vaccination: VaccinationBase,
    session: SessionDep,
    pet: PetDep
):
    new_vaccination = Vaccination(**vaccination.model_dump())
    session.add(new_vaccination)
    await session.commit()
    await session.refresh(new_vaccination)
    return new_vaccination


@app.get("/pets/{pet_id}/vaccinations", response_model=List[VaccinationResponse])
async def get_pet_vaccinations(pet: PetDep):
    return pet.vaccinations


# Напоминания
@app.post("/reminders", response_model=ReminderResponse)
async def create_reminder(
    reminder: ReminderBase,
    session: SessionDep,
    pet: PetDep
):
    new_reminder = Reminder(**reminder.model_dump())
    session.add(new_reminder)
    await session.commit()
    await session.refresh(new_reminder)
    return new_reminder


@app.get("/pets/{pet_id}/reminders", response_model=List[ReminderResponse])
async def get_pet_reminders(pet: PetDep):
    return pet.reminders


# График кормления
@app.post("/feeding-schedules", response_model=FeedingScheduleResponse)
async def create_feeding_schedule(
    schedule: FeedingScheduleBase,
    session: SessionDep,
    pet: PetDep
):
    new_schedule = FeedingSchedule(**schedule.model_dump())
    session.add(new_schedule)
    await session.commit()
    await session.refresh(new_schedule)
    return new_schedule


@app.get("/pets/{pet_id}/feeding-schedules", response_model=List[FeedingScheduleResponse])
async def get_pet_feeding_schedules(pet: PetDep):
    return pet.feeding_schedules
