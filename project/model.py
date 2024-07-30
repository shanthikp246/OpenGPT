from pydantic import BaseModel
from typing import List

class Matrix(BaseModel):
    matrix: List[List[float]]