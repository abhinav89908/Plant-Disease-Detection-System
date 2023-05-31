import numpy as np
from fastapi import FastAPI, File, UploadFile
import random
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model('momdel')
CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

@app.get('/ping')
async def ping():
    return "Hello! I am Abhinav"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    np.argmax(predictions[0])

    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': confidence
    }


    return image


@app.get('/predict')
async def predict(
  file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    return

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port = 8000)
