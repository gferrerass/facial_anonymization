# Z-Image - ComfyUI Workflow Script

Script automatizado para ejecutar un flujo de generación de imágenes con ComfyUI.

## Requisitos previos

- Python 3.10 o superior instalado en tu sistema
- Los modelos descargados (ver sección "Modelos necesarios")
- Git instalado (solo si ComfyUI no esta descargado)

## Estructura de carpetas

```
zimage/
├── main2.py
├── requirements.txt
├── setup.py
├── run.py
├── README.md
├── ComfyUI/              ← Se descarga automaticamente si no existe
├── models/
│   ├── text_encoders/      ← Coloca qwen_3_4b.safetensors aquí
│   ├── unet/               ← Coloca z_image_turbo_bf16.safetensors aquí
│   ├── vae/                ← Coloca ae.safetensors aquí
│   └── checkpoints/
└── output/                 ← Las imágenes generadas se guardan aquí
```

## Modelos necesarios

⚠️ **IMPORTANTE:** Antes de ejecutar, descarga estos modelos y colócalos en las carpetas indicadas:

### 1. Qwen 3.4B (CLIP Model - Procesamiento de texto)
- Tamaño: ~7GB
- Coloca en: `models/text_encoders/`
- Nombre: `qwen_3_4b.safetensors`

### 2. Z-Image Turbo (UNET - Generación)
- Tamaño: ~2.5GB
- Coloca en: `models/unet/`
- Nombre: `z_image_turbo_bf16.safetensors`

### 3. AE VAE (Codificador/Decodificador)
- Tamaño: ~200MB
- Coloca en: `models/vae/`
- Nombre: `ae.safetensors`

## Instalación rápida

### Paso 1: Descargar modelos

Descarga los tres modelos mencionados arriba y colócalos en sus carpetas respectivas dentro de `zimage/models/`.

### Paso 2: Configurar el entorno (una sola vez)

**En cualquier sistema (Windows, Linux, macOS):**
```bash
cd zimage
python setup.py
```

Si ComfyUI no existe, el setup lo descargara en `zimage/ComfyUI`.

### Paso 3: Ejecutar

**En cualquier sistema (Windows, Linux, macOS):**
```bash
python run.py
```

Las imágenes generadas aparecerán en la carpeta `output/`

## Archivos incluidos

- **main2.py** - Script principal que ejecuta el flujo de generación
- **setup.py** - Script de configuración inicial (crea venv e instala dependencias)
- **run.py** - Script para ejecutar el workflow
- **requirements.txt** - Dependencias de Python necesarias
- **models/** - Carpeta para almacenar los modelos localmente
- **output/** - Carpeta donde se guardan las imágenes generadas

## Cómo funciona

El script `main2.py` ejecuta el siguiente flujo:

1. **Configuración de rutas:** Prepara las carpetas locales de `models/` y `output/`
2. **Carga de modelos:**
   - CLIP Model (procesamiento de texto)
   - UNET (generación de imágenes)
   - VAE (codificador/decodificador)
3. **Procesamiento:**
   - Convierte tu prompt de texto en embedding
   - Genera imagen latente
   - Aplica denoising con 9 pasos
   - Decodifica a imagen final
4. **Salida:** Guarda la imagen en `output/` con prefijo personalizable

## Personalización

Puedes modificar `main2.py` para cambiar:

- **Prompt de imagen:** Línea ~137
  ```python
  text="Tu prompt aquí"
  ```Busca `text="Latina female..."`
  ```python
  text="Tu descripción de imagen aquí"
  ```

- **Tamaño de imagen:**
  ```python
  width=1024, height=1024  # Cambia a 512, 768, etc.
  ```

- **Pasos de generación:**
  ```python
  steps=9  # Aumenta para más calidad (15-20 es excelente)
  ```

- **Prefix del archivo de salida:**
  ```python
  filename_prefix="z-image"  # Cambia a un nombre personalizado
  ```
### Error: "ComfyUI not found"
Asegúrate de que la carpeta `ComfyUI` está en el mismo nivel que `zimage`.

### Error: "ModelSamplingAuraFlow not found"
Actualiza ComfyUI - este es un nodo base en versiones recientes.

### Memoria insuficiente (CUDA/GPU)
- Reduce el tamaño de imagen: `width=512, height=512`
- Reduce los pasos: `steps=6`
- Cambia a CPU (más lento)
❌ Error: "ComfyUI not found"
**Causa:** El script no puede encontrar la carpeta ComfyUI  
**Solución:** Asegúrate de que la carpeta `ComfyUI` está en el mismo nivel que `zimage`

### ❌ Error: "Model not found" (qwen_3_4b.safetensors, etc.)
**Causa:** Los modelos no están en las carpetas correctas  
**Solución:** 
- Verifica que descargaste todos los 3 modelos
- Coloca cada uno en su carpeta correspondiente dentro de `models/`
- Los nombres deben ser exactos (mayúsculas/minúsculas)

### ❌ Error de memoria (CUDA out of memory / RAM full)
**Causa:** El sistema no tiene suficiente memoria GPU/CPU  
**Solución:**
- Reduce el tamaño: `width=512, height=512`
- Reduce los pasos: `steps=6`

### ❌ El script se congela al iniciar
**Causa:** ComfyUI está cargando los modelos grandes (puede tomar varios minutos)  
**Solución:** Espera, los modelos de 7GB+ tardan en cargar la primera vez

### ❌ No se generan imágenes / carpeta output vacía
**Causa:** Los modelos no cargaron correctamente o hubo un error silencioso  
**Solución:**
- Verifica que los modelos están en las carpetas correctas
- Abre la terminal para ver los mensajes de error
- Comprueba que tienes suficiente espacio en disco