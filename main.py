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

    # -------------------------
    # 1. PRIORIDADE: CPF perto da palavra CPF
    # -------------------------
    match = re.search(
        r"CPF.{0,120}?(\d{3}\.\d{3}\.\d{3}-\d{2})",
        texto_limpo,
        re.IGNORECASE
    )

    if match:
        candidato = match.group(1)
        if cpf_valido(candidato):
            return candidato

    # -------------------------
    # 2. FALLBACK: qualquer CPF formatado válido
    # -------------------------
    possiveis = re.findall(r"\d{3}\.\d{3}\.\d{3}-\d{2}", texto_limpo)

    for cpf in possiveis:
        if cpf_valido(cpf):
            return cpf

    return None


def preparar_variacoes(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")

    variacoes = []

    for angulo in [0, 90, 180, 270]:
        img = image.rotate(angulo, expand=True)

        w, h = img.size
        img = img.resize((w * 2, h * 2))

        variacoes.append(img)

        gray = ImageOps.grayscale(img)
        variacoes.append(gray)

        contrast = ImageEnhance.Contrast(gray).enhance(2.5)
        variacoes.append(contrast)

        sharp = contrast.filter(ImageFilter.SHARPEN)
        variacoes.append(sharp)

        binary_150 = sharp.point(lambda p: 255 if p > 150 else 0)
        variacoes.append(binary_150)

        binary_120 = sharp.point(lambda p: 255 if p > 120 else 0)
        variacoes.append(binary_120)

    return variacoes


def fazer_ocr_para_cpf(image: Image.Image):
    textos = []

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 12",
        "--oem 3 --psm 4",
    ]

    for img in preparar_variacoes(image):
        for config in configs:
            try:
                texto = pytesseract.image_to_string(
                    img,
                    lang="por",
                    config=config
                )

                if texto.strip():
                    textos.append(texto.strip())

                    cpf = extrair_cpf(texto)

                    if cpf:
                        return cpf, "\n\n".join(textos)

            except Exception:
                continue

    texto_final = "\n\n".join(textos)
    return None, texto_final


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

        cpf, texto_ocr = fazer_ocr_para_cpf(image)

        if cpf:
            return {
                "cpf": cpf,
                "encontrado": True,
                "mensagem": "cpf encontrado",
                "texto_completo": texto_ocr[:3000]
            }

        return {
            "cpf": None,
            "encontrado": False,
            "mensagem": "cpf não encontrado",
            "texto_completo": texto_ocr[:3000]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
