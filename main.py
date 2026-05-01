from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from io import BytesIO
import re

app = FastAPI(title="CPF OCR API")


class ImageRequest(BaseModel):
    image_url: str


def cpf_valido(cpf: str) -> bool:
    cpf = re.sub(r"\D", "", cpf)

    if len(cpf) != 11:
        return False

    if cpf == cpf[0] * 11:
        return False

    soma = sum(int(cpf[i]) * (10 - i) for i in range(9))
    digito1 = (soma * 10) % 11
    if digito1 == 10:
        digito1 = 0

    soma = sum(int(cpf[i]) * (11 - i) for i in range(10))
    digito2 = (soma * 10) % 11
    if digito2 == 10:
        digito2 = 0

    return digito1 == int(cpf[9]) and digito2 == int(cpf[10])


def formatar_cpf(cpf: str):
    cpf = re.sub(r"\D", "", cpf)

    if len(cpf) != 11:
        return None

    return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"


def normalizar_texto(texto: str) -> str:
    texto = texto.replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def extrair_cpf(texto: str):
    texto_limpo = texto.replace("\n", " ")
    texto_limpo = re.sub(r"\s+", " ", texto_limpo)

    padroes = [
        r"(\d{3}\s*[\.,]\s*\d{3}\s*[\.,]\s*\d{3}\s*[-–—]\s*\d{2})",
        r"(\d{3}\s+\d{3}\s+\d{3}\s+[-–—]?\s*\d{2})",
    ]

    for padrao in padroes:
        encontrados = re.findall(padrao, texto_limpo)

        for candidato in encontrados:
            cpf_formatado = formatar_cpf(candidato)

            if cpf_formatado and cpf_valido(cpf_formatado):
                return cpf_formatado

    return None

def preparar_variacoes(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")

    variacoes = []

    for angulo in [0, 90, 180, 270]:
        img = image.rotate(angulo, expand=True)

        w, h = img.size
        img = img.resize((w * 2, h * 2))

        gray = ImageOps.grayscale(img)
        contrast = ImageEnhance.Contrast(gray).enhance(2.5)
        sharp = contrast.filter(ImageFilter.SHARPEN)

        variacoes.append(sharp)

        # Recortes para tentar isolar áreas pequenas do documento
        w, h = sharp.size

        recortes = [
            sharp.crop((0, 0, w, h)),  # imagem inteira
            sharp.crop((0, int(h * 0.25), w, int(h * 0.75))),  # faixa central
            sharp.crop((int(w * 0.20), int(h * 0.20), int(w * 0.80), int(h * 0.80))),  # centro
            sharp.crop((0, int(h * 0.40), w, h)),  # metade inferior
            sharp.crop((int(w * 0.25), int(h * 0.35), int(w * 0.75), int(h * 0.90))),  # centro inferior
        ]

        variacoes.extend(recortes)

    return variacoes


def fazer_ocr_para_cpf(image: Image.Image):
    configs = [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,-–— ",
        "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.,-–— ",
    ]

    for img in preparar_variacoes(image):
        for config in configs:
            try:
                texto = pytesseract.image_to_string(
                    img,
                    lang="por",
                    config=config
                )

                cpf = extrair_cpf(texto)

                if cpf:
                    return cpf, ""

            except Exception:
                continue

    return None, ""

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "cpf-ocr-api"
    }


@app.post("/ler-imagem")
def ler_imagem(payload: ImageRequest):
    try:
        response = requests.get(payload.image_url, timeout=30)

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail="Nao foi possivel baixar a imagem"
            )

        image = Image.open(BytesIO(response.content))

        cpf, _ = fazer_ocr_para_cpf(image)

        if cpf:
            return {
                "cpf": cpf,
                "encontrado": True,
                "mensagem": "cpf encontrado"
            }

        return {
            "cpf": None,
            "encontrado": False,
            "mensagem": "cpf não encontrado"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
