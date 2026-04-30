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


def formatar_cpf(cpf: str) -> str:
    cpf = re.sub(r"\D", "", cpf)

    if len(cpf) != 11:
        return None

    return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"


def extrair_cpf(texto: str):
    texto_limpo = texto.replace("\n", " ")
    texto_limpo = re.sub(r"\s+", " ", texto_limpo)

    # CPF formatado ou sem pontuação
    padroes = [
        r"\d{3}\.\d{3}\.\d{3}-\d{2}",
        r"\d{3}\s*\d{3}\s*\d{3}\s*\d{2}",
        r"\d{11}"
    ]

    for padrao in padroes:
        encontrados = re.findall(padrao, texto_limpo)
        for item in encontrados:
            cpf = formatar_cpf(item)
            if cpf:
                return cpf

    return None


def preparar_variacoes(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")

    variacoes = []

    # Testa rotações porque fotos de documento podem vir de lado
    for angulo in [0, 90, 180, 270]:
        img = image.rotate(angulo, expand=True)

        # aumenta imagem
        w, h = img.size
        img = img.resize((w * 2, h * 2))

        # original ampliada
        variacoes.append(img)

        # escala de cinza
        gray = ImageOps.grayscale(img)
        variacoes.append(gray)

        # contraste
        contrast = ImageEnhance.Contrast(gray).enhance(2.5)
        variacoes.append(contrast)

        # nitidez
        sharp = contrast.filter(ImageFilter.SHARPEN)
        variacoes.append(sharp)

        # binarização clara
        binary_150 = sharp.point(lambda p: 255 if p > 150 else 0)
        variacoes.append(binary_150)

        # binarização mais forte
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
                pass

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
