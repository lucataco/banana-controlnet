import sys
import requests
import base64
from io import BytesIO
from PIL import Image
import banana_dev as banana

img_name = sys.argv[1:][0]
with open(img_name, "rb") as f:
    bytes = f.read()
    encoded = base64.b64encode(bytes).decode('utf-8')

model_inputs = {
    'prompt': 'rihanna best quality, extremely detailed',
    'negative_prompt': 'monochrome, lowres, bad anatomy, worst quality, low quality',
    'num_inference_steps': 20,
    'image_data': encoded
}

api_key=""
model_key=""
res = banana.run(api_key, model_key, model_inputs)
out = res["modelOutputs"][0]

image_byte_string = out["canny_base64"]
image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("edge.jpg")

image_byte_string = out["image_base64"]
image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")

#737.268