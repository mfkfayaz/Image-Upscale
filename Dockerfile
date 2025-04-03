FROM python:3.8

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy repository
COPY . /app/

# Install Python requirements
RUN pip install -r requirements.txt
RUN pip install flask gunicorn

# Download pre-trained models
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights/

# Expose port
EXPOSE 8080

# Start the Flask app
CMD gunicorn --bind 0.0.0.0:$PORT app:app
