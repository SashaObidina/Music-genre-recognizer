import fastapi
import common
from load_m import *

model, scaler, style = get_model()

# REST server definition
app = fastapi.FastAPI()

# inference method
@app.post('/music/')
def get_style(X: common.queryMessage) -> common.ResponseStyle:
    y, sr = X.params()
    return common.ResponseStyle(style=get_style_(y, sr, model, scaler, style))