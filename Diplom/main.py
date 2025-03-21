# Импорт необходимых модулей
from typing import List, Optional, Literal
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime,
    ForeignKey, Boolean, Table, select, delete, func
)
from sqlalchemy.orm import relationship, declarative_base, selectinload
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
    name = Column(String(50))
    type = Column(String(50))
    breed = Column(String(50), nullable=True)
    birth_date = Column(Date, nullable=True)
    gender = Column(String(10))

    health_records = relationship("HealthRecord", back_populates="pet")
    vaccinations = relationship("Vaccination", back_populates="pet")
    reminders = relationship("Reminder", back_populates="pet")
    feeding_schedules = relationship("FeedingSchedule", back_populates="pet")
    users = relationship("User", secondary=user_pet_association, back_populates="pets")


class PetBase(BaseModel):
    name: str = Field(..., max_length=50)
    type: str = Field(..., max_length=50)
    breed: Optional[str] = Field(None, max_length=50)
    birth_date: date = Field(...)  # Теперь обязательное поле
    gender: Literal["male", "female", "other"]

    @field_validator('birth_date')
    def validate_birth_date(cls, value):
        if value > date.today():
            raise ValueError("Дата рождения не может быть в будущем")
        return value


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


class PetResponse(BaseModel):
    id: int
    name: str
    type: str
    breed: Optional[str]
    birth_date: Optional[date]
    gender: str

    @computed_field
    @property
    def months(self) -> Optional[int]:
        if self.birth_date is None:
            return None
        return calculate_age(self.birth_date)


class PetResponseExtended(PetResponse):
    last_weight: Optional[float] = None
    active_reminders_count: int = 0
    vaccinations_count: int = 0
    owners_count: int = 0

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


class HealthRecordResponse(BaseModel):
    id: int
    record_date: date
    weight: float
    description: str

    class Config:
        from_attributes = True


class VaccinationResponse(BaseModel):
    id: int
    name: str
    date_administered: date
    next_date: Optional[date]
    repeated: bool

    class Config:
        from_attributes = True


class FeedingScheduleResponse(BaseModel):
    id: int
    pet_id: int
    feeding_time: datetime
    food_type: str
    quantity: str
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class PetUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    breed: Optional[str] = None
    birth_date: Optional[date] = None
    gender: Optional[Literal["male", "female", "other"]] = None
    age: Optional[int] = None


class HealthRecordCreate(BaseModel):
    weight: float
    description: str


class VaccinationCreate(BaseModel):
    name: str
    date_administered: date
    next_date: Optional[date] = None
    repeated: bool = False


class FeedingScheduleCreate(BaseModel):
    feeding_time: datetime
    food_type: str
    quantity: str
    notes: Optional[str] = None


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


@field_validator('months')
def validate_months(cls, value):
    if value is not None and value < 0:
        raise ValueError("Возраст не может быть отрицательным")
    return value


@app.post("/pets", response_model=PetResponse)
async def create_pet(
    pet: PetBase,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_user)]
):
    try:
        db_pet = Pet(
            name=pet.name,
            type=pet.type,
            breed=pet.breed,
            birth_date=pet.birth_date,
            gender=pet.gender
        )

        session.add(db_pet)
        await session.flush()

        stmt = user_pet_association.insert().values(
            user_id=current_user.id,
            pet_id=db_pet.id
        )
        await session.execute(stmt)

        await session.commit()
        await session.refresh(db_pet)
        return db_pet

    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=400,
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


@app.get("/pets", response_model=List[PetResponseExtended])
async def get_user_pets(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_user)]
):
    # Подзапрос для последнего веса с явной корреляцией
    last_weight_subquery = (
        select(HealthRecord.weight)
        .where(HealthRecord.pet_id == Pet.id)
        .order_by(HealthRecord.record_date.desc())
        .limit(1)
        .correlate(Pet)  # Явно указываем корреляцию
        .scalar_subquery()
        .label("last_weight")
    )

    stmt = (
        select(
            Pet,
            last_weight_subquery,
            func.count(Reminder.id).filter(Reminder.is_completed == False).label("active_reminders_count"),
            func.count(Vaccination.id).label("vaccinations_count"),
            func.count(user_pet_association.c.user_id).label("owners_count")
        )
        .join(user_pet_association, Pet.id == user_pet_association.c.pet_id)
        .outerjoin(Reminder, Pet.id == Reminder.pet_id)
        .outerjoin(Vaccination, Pet.id == Vaccination.pet_id)
        .where(user_pet_association.c.user_id == current_user.id)
        .group_by(Pet.id)
    )

    result = await session.execute(stmt)
    pets = result.all()

    return [
        {
            "id": pet.Pet.id,
            "name": pet.Pet.name,
            "type": pet.Pet.type,
            "breed": pet.Pet.breed,
            "birth_date": pet.Pet.birth_date,
            "gender": pet.Pet.gender,
            "last_weight": pet.last_weight,
            "active_reminders_count": pet.active_reminders_count,
            "vaccinations_count": pet.vaccinations_count,
            "owners_count": pet.owners_count
        }
        for pet in pets
    ]


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


async def check_pet_access(pet_id: int, user_id: int, session: AsyncSession):
    result = await session.execute(
        select(user_pet_association)
        .where(user_pet_association.c.user_id == user_id)
        .where(user_pet_association.c.pet_id == pet_id)
    )
    if not result.scalar():
        raise HTTPException(status_code=403, detail="Нет доступа к питомцу")


@app.get("/pets/{pet_id}", response_model=PetResponse)
async def get_pet(
        pet_id: int,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    await check_pet_access(pet_id, current_user.id, session)

    # Явная загрузка связанных данных
    result = await session.execute(
        select(Pet)
        .where(Pet.id == pet_id)
        .options(
            selectinload(Pet.health_records),
            selectinload(Pet.vaccinations),
            selectinload(Pet.feeding_schedules)
        )
    )
    pet = result.scalar()

    if not pet:
        raise HTTPException(status_code=404, detail="Питомец не найден")
    return pet


@app.get("/pets/{pet_id}/health-records", response_model=List[HealthRecordResponse])
async def get_health_records(
        pet_id: int,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    await check_pet_access(pet_id, current_user.id, session)
    result = await session.execute(select(HealthRecord).where(HealthRecord.pet_id == pet_id))
    return result.scalars().all()


@app.get("/pets/{pet_id}/vaccinations", response_model=List[VaccinationResponse])
async def get_vaccinations(
        pet_id: int,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    await check_pet_access(pet_id, current_user.id, session)
    result = await session.execute(select(Vaccination).where(Vaccination.pet_id == pet_id))
    return result.scalars().all()


@app.get("/pets/{pet_id}/feeding-schedules", response_model=List[FeedingScheduleResponse])
async def get_feeding_schedules(
        pet_id: int,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    await check_pet_access(pet_id, current_user.id, session)
    result = await session.execute(select(FeedingSchedule).where(FeedingSchedule.pet_id == pet_id))
    return result.scalars().all()


@app.get("/pets/{pet_id}/reminders", response_model=List[ReminderResponse])
async def get_pet_reminders(
        pet_id: int,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    await check_pet_access(pet_id, current_user.id, session)
    result = await session.execute(select(Reminder).where(Reminder.pet_id == pet_id))
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


@app.put("/pets/{pet_id}", response_model=PetResponse)
async def update_pet(
        pet_id: int,
        pet_data: PetUpdate,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    await check_pet_access(pet_id, current_user.id, session)

    pet = await session.get(Pet, pet_id)
    if not pet:
        raise HTTPException(status_code=404, detail="Питомец не найден")

    for field, value in pet_data.model_dump(exclude_unset=True).items():
        setattr(pet, field, value)

    await session.commit()
    await session.refresh(pet)
    return pet


@app.post("/pets/{pet_id}/health-records", response_model=HealthRecordResponse)
async def create_health_record(
        pet_id: int,
        record_data: HealthRecordCreate,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    await check_pet_access(pet_id, current_user.id, session)

    new_record = HealthRecord(**record_data.model_dump(), pet_id=pet_id)
    session.add(new_record)
    await session.commit()
    await session.refresh(new_record)
    return new_record


def calculate_age(birth_date: date) -> int:
    if birth_date is None:
        return None
    today = date.today()
    months = (today.year - birth_date.year) * 12 + (today.month - birth_date.month)
    if today.day < birth_date.day:
        months -= 1
    return months


@app.post("/pets/{pet_id}/vaccinations", response_model=VaccinationResponse)
async def create_vaccination(
        pet_id: int,
        vax_data: VaccinationCreate,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    await check_pet_access(pet_id, current_user.id, session)

    new_vax = Vaccination(**vax_data.model_dump(), pet_id=pet_id)
    session.add(new_vax)
    await session.commit()
    await session.refresh(new_vax)
    return new_vax


@app.post("/pets/{pet_id}/feeding-schedules", response_model=FeedingScheduleResponse)
async def create_feeding_schedule(
        pet_id: int,
        feeding_data: FeedingScheduleCreate,
        session: SessionDep,
        current_user: Annotated[User, Depends(get_current_user)]
):
    await check_pet_access(pet_id, current_user.id, session)

    new_feeding = FeedingSchedule(**feeding_data.model_dump(), pet_id=pet_id)
    session.add(new_feeding)
    await session.commit()
    await session.refresh(new_feeding)
    return new_feeding


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
