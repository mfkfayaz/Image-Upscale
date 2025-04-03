import os
import sys
import importlib
from flask import Flask, request, send_file, render_template_string
import torch
import torchvision

# Monkey patch for compatibility with newer torchvision versions
if not hasattr(torchvision.transforms, 'functional_tensor'):
    # Create a compatibility module
    import types
    from torchvision.transforms import functional as F
    
    # Add rgb_to_grayscale function to the module
    functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
    functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
    
    # Add the module to sys.modules
    sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor

# Now import the Real-ESRGAN modules
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = Flask(__name__)

# Set up model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path="weights/RealESRGAN_x4plus.pth",
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=torch.cuda.is_available()
)

@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-ESRGAN Image Upscaler</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { margin-top: 20px; }
            .btn { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Real-ESRGAN Image Upscaler</h1>
        <div class="container">
            <form action="/upscale" method="post" enctype="multipart/form-data">
                <p>Select an image to upscale (4x):</p>
                <input type="file" name="image" accept="image/*" required>
                <button type="submit" class="btn">Upscale Image</button>
            </form>
        </div>
    </body>
    </html>
    """)

@app.route("/upscale", methods=["POST"])
def upscale():
    if "image" not in request.files:
        return "No image uploaded", 400
        
    file = request.files["image"]
    input_path = "input.jpg"
    output_path = "output.jpg"
    
    file.save(input_path)
    
    # Upscale the image
    try:
        upsampler.enhance(input_path, outpath=output_path)
        return send_file(output_path, mimetype="image/jpeg")
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
