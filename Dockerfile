FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
       python3-pip python3-dev build-essential \
       libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.1/compat/

RUN python3 -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    REMBG_MODEL=u2net

CMD ["python3", "-u", "handler.py"]
