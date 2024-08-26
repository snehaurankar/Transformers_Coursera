from transformers import pipeline
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

gpt2_model_dir = r'C:\Users\A200155742\PycharmProjects\Coursera Fast API\gpt2'
distilbert_model_dir = r'C:\Users\A200155742\PycharmProjects\Coursera Fast API\distilbert-base-uncased'

generator = pipeline('text-generation', model=gpt2_model_dir, tokenizer=gpt2_model_dir)
sentiment_analysis = pipeline('sentiment-analysis', model=distilbert_model_dir, tokenizer=distilbert_model_dir)

templates = Jinja2Templates(directory=".")

app = FastAPI()

class Body(BaseModel):
    text: str

@app.get('/')
def root():
    return Response("<h1>A self documenting API to interact with a GPT2 model and generate text</h1>")

@app.get('/generate', response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("transformer_html.html", {'request': request})

@app.post('/predict')
async def predict(request: Request, body: Body):
    results = generator(body.text, max_length=35, num_return_sequences=1)
    return results[0].get('generated_text')

@app.get('/lab')
def lab():
    return {"request": "GET"}

@app.get('/analyze', response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("transformer_html.html", {'request': request})

@app.post('/sentiment')
async def sentiment(request: Request, body: Body):
    results = sentiment_analysis(body.text, max_length=35)
    return results[0]