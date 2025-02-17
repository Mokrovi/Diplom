from typing import List, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Boolean, Table, select
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from fastapi import FastAPI, Depends, HTTPException, status, Response
from datetime import date, datetime
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

Base = declarative_base()
app = FastAPI()
engine = create_async_engine('sqlite+aiosqlite:///Diplom.db')
new_session = async_sessionmaker(engine, expire_on_commit=False)

# Конфигурация JWT
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


Base = declarative_base()
app = FastAPI()
engine = create_async_engine('sqlite+aiosqlite:///Diplom.db')
new_session = async_sessionmaker(engine, expire_on_commit=False)

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
    password = Column(String(50))

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


# schemas

# User schemas
class UserBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., max_length=100)
    password: str = Field(..., min_length=6, max_length=50)


class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True


# Pet schemas
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


# Health Record schemas
class HealthRecordBase(BaseModel):
    record_date: date = Field(default_factory=date.today)  # Переименовано в record_date
    weight: float = Field(..., gt=0)
    description: str = Field(..., max_length=500)


class HealthRecordResponse(HealthRecordBase):
    id: int
    pet_id: int

    class Config:
        from_attributes = True


# Vaccination schemas
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


# Reminder schemas
class ReminderBase(BaseModel):
    description: str = Field(..., max_length=200)
    reminder_date: datetime
    is_completed: bool = False


class ReminderResponse(ReminderBase):
    id: int
    pet_id: int

    class Config:
        from_attributes = True


# Feeding Schedule schemas
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


# Requests

@app.post("/rebuild_bd")
async def rebuild_bd():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    return {"ok": True}


@app.post("/post_user", response_model=UserResponse)
async def post_user(
        data: UserBase,
        session: AsyncSession = Depends(get_session)
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


# User Endpoints
@app.post("/users", response_model=UserResponse)
async def create_user(user: UserBase, session: AsyncSession = Depends(get_session)):
    # Проверка уникальности email
    existing_user = await session.execute(select(User).where(User.email == user.email))
    if existing_user.scalar():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    new_user = User(**user.model_dump())
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user


@app.get("/users", response_model=List[UserResponse])
async def get_all_users(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(User))
    users = result.scalars().all()
    return users


# Pet Endpoints
@app.post("/pets", response_model=PetResponse)
async def create_pet(pet: PetBase, session: AsyncSession = Depends(get_session)):
    # Проверка существования владельца
    owner = await session.get(User, pet.user_id)
    if not owner:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    new_pet = Pet(**pet.model_dump())
    session.add(new_pet)
    await session.commit()
    await session.refresh(new_pet)
    return new_pet


@app.get("/pets", response_model=List[PetResponse])
async def get_all_pets(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Pet))
    pets = result.scalars().all()
    return pets


# Health Record Endpoints
@app.post("/health-records", response_model=HealthRecordResponse)
async def create_health_record(
        record: HealthRecordBase,
        session: AsyncSession = Depends(get_session)
):
    # Проверка существования питомца
    pet = await session.get(Pet, record.pet_id)
    if not pet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pet not found"
        )

    new_record = HealthRecord(**record.model_dump())
    session.add(new_record)
    await session.commit()
    await session.refresh(new_record)
    return new_record


@app.get("/pets/{pet_id}/health-records", response_model=List[HealthRecordResponse])
async def get_pet_health_records(
        pet_id: int,
        session: AsyncSession = Depends(get_session)
):
    pet = await session.get(Pet, pet_id)
    if not pet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pet not found"
        )

    result = await session.execute(
        select(HealthRecord).where(HealthRecord.pet_id == pet_id)
    )
    return result.scalars().all()


# Vaccination Endpoints
@app.post("/vaccinations", response_model=VaccinationResponse)
async def create_vaccination(
        vaccination: VaccinationBase,
        session: AsyncSession = Depends(get_session)
):
    pet = await session.get(Pet, vaccination.pet_id)
    if not pet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pet not found"
        )

    new_vaccination = Vaccination(**vaccination.model_dump())
    session.add(new_vaccination)
    await session.commit()
    await session.refresh(new_vaccination)
    return new_vaccination


@app.get("/pets/{pet_id}/vaccinations", response_model=List[VaccinationResponse])
async def get_pet_vaccinations(
        pet_id: int,
        session: AsyncSession = Depends(get_session)
):
    pet = await session.get(Pet, pet_id)
    if not pet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pet not found"
        )

    result = await session.execute(
        select(Vaccination).where(Vaccination.pet_id == pet_id)
    )
    return result.scalars().all()


# Reminder Endpoints
@app.post("/reminders", response_model=ReminderResponse)
async def create_reminder(
        reminder: ReminderBase,
        session: AsyncSession = Depends(get_session)
):
    pet = await session.get(Pet, reminder.pet_id)
    if not pet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pet not found"
        )

    new_reminder = Reminder(**reminder.model_dump())
    session.add(new_reminder)
    await session.commit()
    await session.refresh(new_reminder)
    return new_reminder


@app.get("/pets/{pet_id}/reminders", response_model=List[ReminderResponse])
async def get_pet_reminders(
        pet_id: int,
        session: AsyncSession = Depends(get_session)
):
    pet = await session.get(Pet, pet_id)
    if not pet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pet not found"
        )

    result = await session.execute(
        select(Reminder).where(Reminder.pet_id == pet_id)
    )
    return result.scalars().all()


# Feeding Schedule Endpoints
@app.post("/feeding-schedules", response_model=FeedingScheduleResponse)
async def create_feeding_schedule(
        schedule: FeedingScheduleBase,
        session: AsyncSession = Depends(get_session)
):
    pet = await session.get(Pet, schedule.pet_id)
    if not pet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pet not found"
        )

    new_schedule = FeedingSchedule(**schedule.model_dump())
    session.add(new_schedule)
    await session.commit()
    await session.refresh(new_schedule)
    return new_schedule


@app.get("/pets/{pet_id}/feeding-schedules", response_model=List[FeedingScheduleResponse])
async def get_pet_feeding_schedules(
        pet_id: int,
        session: AsyncSession = Depends(get_session)
):
    pet = await session.get(Pet, pet_id)
    if not pet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pet not found"
        )

    result = await session.execute(
        select(FeedingSchedule).where(FeedingSchedule.pet_id == pet_id)
    )
    return result.scalars().all()