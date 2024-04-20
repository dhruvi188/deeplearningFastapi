from pydantic import BaseModel


class DiseaseInfo(BaseModel):
    disease_name: str
    description: str
    possible_steps: str
    image_url: str


class SupplementInfo(BaseModel):
    supplement_name: str
    supplement_image: str
    buy_link: str
