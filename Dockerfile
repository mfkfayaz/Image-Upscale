FROM python:3.8

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
COPY . /app/

# Install Python requirements
RUN pip install -r requirements.txt
RUN pip install flask gunicorn

# Download pre-trained models
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights/

# Create a simple Flask app for the web interface
RUN echo 'import os\n\
from flask import Flask, request, send_file, render_template_string\n\
import torch\n\
from basicsr.archs.rrdbnet_arch import RRDBNet\n\
from realesrgan import RealESRGANer\n\
\n\
app = Flask(__name__)\n\
\n\
# Set up model\n\
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)\n\
upsampler = RealESRGANer(\n\
    scale=4,\n\
    model_path="weights/RealESRGAN_x4plus.pth",\n\
    model=model,\n\
    tile=0,\n\
    tile_pad=10,\n\
    pre_pad=0,\n\
    half=torch.cuda.is_available()\n\
)\n\
\n\
@app.route("/")\n\
def index():\n\
    return render_template_string("""\n\
    <!DOCTYPE html>\n\
    <html>\n\
    <head>\n\
        <title>Real-ESRGAN Image Upscaler</title>\n\
        <style>\n\
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }\n\
            .container { margin-top: 20px; }\n\
            .btn { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }\n\
        </style>\n\
    </head>\n\
    <body>\n\
        <h1>Real-ESRGAN Image Upscaler</h1>\n\
        <div class="container">\n\
            <form action="/upscale" method="post" enctype="multipart/form-data">\n\
                <p>Select an image to upscale (4x):</p>\n\
                <input type="file" name="image" accept="image/*" required>\n\
                <button type="submit" class="btn">Upscale Image</button>\n\
            </form>\n\
        </div>\n\
    </body>\n\
    </html>\n\
    """)\n\
\n\
@app.route("/upscale", methods=["POST"])\n\
def upscale():\n\
    if "image" not in request.files:\n\
        return "No image uploaded", 400\n\
        \n\
    file = request.files["image"]\n\
    input_path = "input.jpg"\n\
    output_path = "output.jpg"\n\
    \n\
    file.save(input_path)\n\
    \n\
    # Upscale the image\n\
    try:\n\
        upsampler.enhance(input_path, outpath=output_path)\n\
        return send_file(output_path, mimetype="image/jpeg")\n\
    except Exception as e:\n\
        return f"Error processing image: {str(e)}", 500\n\
\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))\n\
' > app.py

# Expose port
EXPOSE 8080

# Start the Flask app
CMD gunicorn --bind 0.0.0.0:$PORT app:app
