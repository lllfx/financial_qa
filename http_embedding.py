from typing import List
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel


class HttpEmbedding(Embeddings, BaseModel):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return do_embedding(texts)

    def embed_query(self, text: str) -> List[float]:
        return do_embedding([text])[0]


def do_embedding(texts: List[str]) -> List[List[float]]:
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": texts,
        "encoding_format": "float"
    }
    print(payload)
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer sk-eauzuaijutetemhwdvfmrgtjqpyuvxpcadwpzwjqcqdxanec"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise TypeError("错误")
    data = response.json()
    res = []
    for embedding in data["data"]:
        print(embedding)
        res.append(embedding["embedding"])
    return res
