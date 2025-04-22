from flask import Flask, request, send_file
from io import BytesIO
import os

app = Flask(__name__)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    from inference_client import InferenceClient  # Import inside function for better memory management
    
    # Get prompt from request
    data = request.get_json()
    prompt = data.get('prompt', 'Astronaut riding a horse')
    
    # Create client with environment variable
    client = InferenceClient(
        provider="nebius",
        api_key=os.getenv("HF_API_KEY"),
    )
    
    # Generate image
    image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
    
    # Convert image to bytes
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=False)
