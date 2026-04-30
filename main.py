from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
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
    texto_limpo = texto.replace("\n", " ")
    texto_limpo = re.sub(r"\s+", " ", texto_limpo)

    match = re.search(
        r"CPF.{0,100}?(\d{3}\.?\s*\d{3}\.?\s*\d{3}-?\s*\d{2})",
        texto_limpo,
        re.IGNORECASE
    )

    if match:
        return formatar_cpf(match.group(1))

    return None


def normalizar_texto(texto: str) -> str:
    texto = texto.upper()
    texto = texto.replace("Á", "A").replace("À", "A").replace("Ã", "A").replace("Â", "A")
    texto = texto.replace("É", "E").replace("Ê", "E")
    texto = texto.replace("Í", "I")
    texto = texto.replace("Ó", "O").replace("Õ", "O").replace("Ô", "O")
    texto = texto.replace("Ú", "U")
    texto = texto.replace("Ç", "C")
    return texto


def classificar_documento(texto: str):
    t = normalizar_texto(texto)

    palavras_cnh = [
        "CARTEIRA NACIONAL",
        "HABILITACAO",
        "DRIVER LICENSE",
        "PERMISO",
        "CATEGORIA",
        "VALIDADE",
        "DETRAN",
        "SENATRAN",
        "RENACH",
    ]

    palavras_doc = [
        "CPF",
        "RG",
        "IDENTIDADE",
        "REPUBLICA FEDERATIVA",
        "NOME",
        "NASCIMENTO",
        "FILIACAO",
        "DOC IDENTIDADE",
        "ORGAO EMISSOR",
    ]

    pontos_cnh = sum(1 for p in palavras_cnh if p in t)
    pontos_doc = sum(1 for p in palavras_doc if p in t)

    if pontos_cnh >= 2:
        return "cnh"

    if pontos_doc >= 2:
        return "documento_identidade"

    if pontos_cnh + pontos_doc >= 2:
        return "documento_possivel"

    return "imagem_invalida"


def preparar_imagem(image: Image.Image) -> list[Image.Image]:
    image = image.convert("RGB")

    # Corrige orientação EXIF, comum em foto de celular
    image = ImageOps.exif_transpose(image)

    imagens = []

    # Versão original aumentada
    w, h = image.size
    scale = 2
    img_grande = image.resize((w * scale, h * scale))
    imagens.append(img_grande)

    # Cinza + contraste
    gray = ImageOps.grayscale(img_grande)
    contraste = ImageEnhance.Contrast(gray).enhance(2.0)
    imagens.append(contraste)

    # Binarizada
    binaria = contraste.point(lambda p: 255 if p > 150 else 0)
    imagens.append(binaria)

    return imagens


def fazer_ocr(image: Image.Image) -> str:
    textos = []

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 12",
    ]

    for img in preparar_imagem(image):
        for config in configs:
            try:
                texto = pytesseract.image_to_string(img, lang="por", config=config)
                if texto.strip():
                    textos.append(texto.strip())
            except Exception:
                pass

    return "\n\n".join(textos).strip()


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "document-image-ocr-api"
    }


@app.post("/ler-imagem")
def ler_imagem(payload: ImageRequest):
    try:
        response = requests.get(payload.image_url, timeout=30)

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Nao foi possivel baixar a imagem")

        image = Image.open(BytesIO(response.content))

        texto_limpo = fazer_ocr(image)

        cpf = extrair_cpf(texto_limpo)
        tipo = classificar_documento(texto_limpo)

        # Se achou CPF, aumenta confiança de que é documento
        if cpf and tipo == "imagem_invalida":
            tipo = "documento_possivel"

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
