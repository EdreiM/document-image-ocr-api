from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pytesseract
from PIL import Image
from io import BytesIO
import re

app = FastAPI(title="Document Image OCR API")


class ImageRequest(BaseModel):
    image_url: str


def formatar_cpf(cpf: str) -> str:
    cpf = re.sub(r"\D", "", cpf)

    if len(cpf) != 11:
        return cpf

    return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"


def extrair_cpf(texto: str):
    padrao_formatado = r"\d{3}\.\d{3}\.\d{3}-\d{2}"
    encontrado = re.search(padrao_formatado, texto)

    if encontrado:
        return encontrado.group(0)

    apenas_numeros = re.sub(r"\D", "", texto)

    possiveis = re.findall(r"\d{11}", apenas_numeros)

    for cpf in possiveis:
        return formatar_cpf(cpf)

    return None


def classificar_documento(texto: str):
    texto_upper = texto.upper()

    palavras_documento = [
        "CPF",
        "RG",
        "CARTEIRA NACIONAL DE HABILITAÇÃO",
        "CNH",
        "IDENTIDADE",
        "REPÚBLICA FEDERATIVA DO BRASIL",
        "NOME",
        "DATA DE NASCIMENTO",
        "VALIDADE",
        "DOC IDENTIDADE",
        "ÓRGÃO EMISSOR",
    ]

    pontos = 0

    for palavra in palavras_documento:
        if palavra in texto_upper:
            pontos += 1

    if "CARTEIRA NACIONAL DE HABILITAÇÃO" in texto_upper or "CNH" in texto_upper:
        return "cnh"

    if "CPF" in texto_upper or "IDENTIDADE" in texto_upper or "RG" in texto_upper:
        return "documento_identidade"

    if pontos >= 2:
        return "documento_possivel"

    return "imagem_invalida"


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "document-image-ocr-api"
    }


@app.post("/ler-imagem")
def ler_imagem(payload: ImageRequest):
    try:
        response = requests.get(payload.image_url, timeout=20)

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Nao foi possivel baixar a imagem")

        image = Image.open(BytesIO(response.content))

        texto = pytesseract.image_to_string(image, lang="por")

        texto_limpo = texto.strip()

        cpf = extrair_cpf(texto_limpo)
        tipo = classificar_documento(texto_limpo)

        documento_valido = tipo != "imagem_invalida"

        return {
            "tipo": tipo,
            "cpf": cpf,
            "texto_completo": texto_limpo,
            "documento_valido": documento_valido,
            "mensagem": "documento identificado" if documento_valido else "imagem nao parece documento oficial"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
