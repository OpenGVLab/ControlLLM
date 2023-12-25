import argparse

import uvicorn
from fastapi import Form
from fastapi.responses import JSONResponse, Response
import json

from .tools import *
from cllm.services import app, pool
from ..hf_pipeline import HuggingfacePipelineNLP

parser = argparse.ArgumentParser(description="Image Perception API")
parser.add_argument("--host", type=str, default="localhost", help="Host")
parser.add_argument("--port", type=int, default=10049, help="Port")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()


class RawResponse(Response):
    media_type = "binary/octet-stream"

    def render(self, content: bytes) -> bytes:
        return bytes([b ^ 0x54 for b in content])


@app.post("/question_answering_with_context")
@pool.register(
    lambda: HuggingfacePipelineNLP(
        "question-answering", args.device, model="deepset/roberta-base-squad2"
    )
)
async def question_answering_with_context(
    context: str = Form(...), question: str = Form(...)
):
    model = question_answering_with_context.__wrapped__.model
    output = model({"context": context, "question": question})
    return JSONResponse(output)


@app.post("/text_to_text_generation")
@pool.register(
    lambda: HuggingfacePipelineNLP(
        "text2text-generation", args.device, model="google/flan-t5-base"
    )
)
async def text_to_text_generation(text: str = Form(...)):
    model = text_to_text_generation.__wrapped__.model
    output = model(text)
    return JSONResponse(output)


@app.post("/text_to_tags")
@pool.register(lambda: Text2Tags(args.device))
async def text_to_tags(text: str = Form(...)):
    model = text_to_tags.__wrapped__.model
    output = model(text)
    return JSONResponse(output)


@app.post("/sentiment_analysis")
@pool.register(
    lambda: HuggingfacePipelineNLP(
        device=args.device,
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )
)
async def sentiment_analysis(text: str = Form(...)):
    model = sentiment_analysis.__wrapped__.model
    output = model(text)
    return JSONResponse(output)


@app.post("/summarization")
@pool.register(lambda: HuggingfacePipelineNLP("summarization", device=args.device))
async def summarization(text: str = Form(...)):
    model = summarization.__wrapped__.model
    output = model(text)
    return JSONResponse(output)


@app.post("/get_weather")
@pool.register(lambda: WeatherAPI(device=args.device))
async def get_weather(location: str = Form(...)):
    model = get_weather.__wrapped__.model
    output = model(location)
    return JSONResponse(output)


@app.post("/extract_location")
@pool.register(
    lambda: HuggingfacePipelineNLP(
        "ner",
        device=args.device,
        tokenizer="ml6team/bert-base-uncased-city-country-ner",
        model="ml6team/bert-base-uncased-city-country-ner",
        aggregation_strategy="simple",
    )
)
async def extract_location(text: str = Form(...)):
    model = extract_location.__wrapped__.model
    output = model(text)
    output = json.dumps(output, ensure_ascii=False, default=float)
    output = json.loads(output)
    return JSONResponse(output)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
