# Facial anonymization

Aplicacion de anonimizacion facial basada en ComfyUI. El proyecto permite:

- Generar rostros anonimizados a partir de imagenes en `input/`.
- Evaluar calidad y anonimato con metricas CLIP, LPIPS e InsightFace (opcional).
- Ejecutar un flujo combinado de generacion + evaluacion iterativa.

## Estado actual del repo

Los scripts principales existentes son:

- `setup.py`: prepara entorno virtual, instala dependencias, ComfyUI y custom nodes.
- `generation.py`: solo generacion de imagenes anonimizadas.
- `evaluation.py`: solo evaluacion de un par de imagenes.
- `main.py`: flujo completo (generacion + evaluacion iterativa).
- `shared_utils.py`: utilidades compartidas.

Nota: actualmente no existe `run.py`, `generate.py` ni `evaluate.py` en este repositorio.

## Requisitos

- Python 3.10 o superior.
- Git instalado.
- GPU NVIDIA recomendada para rendimiento (si no, funciona en CPU con menor velocidad).
- Espacio en disco para modelos (recomendado: 50 GB o mas).

## Instalacion

### 1. Clonar y entrar al proyecto

```bash
git clone <repository-url>
cd facial_anonymization
```

### 2. Ejecutar setup

```bash
python setup.py
```

`setup.py` realiza automaticamente:

- Creacion de `venv/`.
- Instalacion de dependencias Python en el entorno virtual.
- Descarga/configuracion de `ComfyUI/`.
- Instalacion de custom nodes requeridos.

Los scripts (`main.py`, `generation.py`, `evaluation.py`) intentan relanzarse dentro de `venv` si detectan que estas fuera del entorno virtual.

## Modelos requeridos

Descarga y coloca estos archivos en las rutas indicadas:

- `models/text_encoders/qwen_3_4b.safetensors`
- `models/unet/z_image_turbo_bf16.safetensors`
- `models/vae/ae.safetensors`
- `models/ultralytics/bbox/face_yolov8m.pt`
- `models/controlnet/Z-Image-Turbo-Fun-Controlnet-Union.safetensors`

## Uso

### Flujo completo (recomendado)

```bash
python main.py
```

Este modo:

- Genera imagen anonima.
- Evalua metricas.
- Ajusta `strength`/`denoise` por iteracion hasta cumplir umbrales o llegar a `--max-iterations`.

Ejemplos:

```bash
python main.py --input input --output output
python main.py --strength 0.7 --denoise 0.6 --max-images 10
python main.py --insightface-threshold 0.65 --clip-threshold 0.75 --lpips-threshold 0.3
```

### Solo generacion

```bash
python generation.py
python generation.py --input input --output output --strength 0.7 --denoise 0.6 --max-images 5
```

### Solo evaluacion

```bash
python evaluation.py input/original.jpg output/original_anonymized_20260305_120000.png
python evaluation.py input/original.jpg output/original_anonymized_20260305_120000.png --no_crop
```

## Parametros CLI

### `main.py` y `generation.py`

- `--input`: carpeta de entrada (default `input`).
- `--output`: carpeta de salida (default `output`).
- `--max-images`: maximo de imagenes a procesar.
- `--strength`: fuerza de ControlNet (default `0.7`).
- `--denoise`: denoise en sampling (default `0.6`).

### Solo `main.py`

- `--insightface-threshold` (default `0.65`).
- `--clip-threshold` (default `0.75`).
- `--lpips-threshold` (default `0.3`).
- `--max-iterations` (default `3`).

### Solo `evaluation.py`

- `original`: ruta a imagen original.
- `anonymized`: ruta a imagen anonimizada.
- `--no_crop`: omite deteccion/crop de rostro (asume imagenes ya recortadas).

## Metricas

- CLIP similarity: similitud semantica global.
- LPIPS distance: distancia perceptual.
- InsightFace distance: distancia en embeddings faciales (si InsightFace esta disponible).

Si InsightFace no esta instalado o no carga, el pipeline sigue funcionando y esa metrica se omite.

## Estructura resumida

```text
facial_anonymization/
|- main.py
|- generation.py
|- evaluation.py
|- shared_utils.py
|- setup.py
|- requirements.txt
|- README.md
|- input/
|- output/
|- models/
|- ComfyUI/
`- venv/
```

## Troubleshooting rapido

- Error de entorno virtual: verifica que exista `venv/` y reejecuta `python setup.py`.
- Error por modelo faltante: confirma nombre exacto y carpeta en `models/`.
- Sin deteccion facial en evaluacion: prueba `--no_crop` si tus imagenes ya estan recortadas.
- Rendimiento bajo: usa GPU CUDA y reduce `--max-images` para pruebas.