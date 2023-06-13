from pydantic import BaseModel
import numpy as np

class ResponseStyle(BaseModel):
    style: str

class queryMessage(BaseModel):
    time_series: list
    sr: int

    def params(self):
        return self.time_series, self.sr
