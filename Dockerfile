# Usar una imagen ligera oficial de Python
FROM python:3.11-slim

# Evitar archivos .pyc y asegurar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear el directorio de trabajo principal
WORKDIR /app

# Crear la estructura de carpetas de datos para que el script no falle
RUN mkdir -p /app/data/raw /app/data/processed /app/docs /app/artifacts

# Copiar e instalar las dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del proyecto al contenedor
COPY . /app/

# Comando predeterminado para ejecutar el pipeline de IA
CMD ["python", "script_entrenamiento.py"]