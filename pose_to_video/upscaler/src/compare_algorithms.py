from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('../dist/model.h5')


# Function to perform AI upscaling
def upscale_ai(image):
    input_img = np.array(image).astype('float32') / 255.0
    input_img = np.expand_dims(input_img, axis=0)
    upscaled_img = model.predict(input_img)[0]
    upscaled_img = (upscaled_img * 255.0).astype('uint8')
    return Image.fromarray(upscaled_img)


# Open the original image and resize it
original = Image.open('../figures/original.png').resize((768, 768))
# Downscale the original image
downscaled = original.resize((256, 256), Image.LANCZOS)

# Upscale using different algorithms
upscaled_nn = downscaled.resize((768, 768), Image.NEAREST)
upscaled_nn.save('../figures/nearest-neighbor.png')

upscaled_bicubic = downscaled.resize((768, 768), Image.BICUBIC)
upscaled_bicubic.save('../figures/bicubic.png')

upscaled_lanczos = downscaled.resize((768, 768), Image.LANCZOS)
upscaled_lanczos.save('../figures/lanczos.png')

upscaled_ai = upscale_ai(downscaled)
upscaled_ai.save('../figures/ai.png')

# Crop and save the images
images = {
    'original': original,
    'nearest-neighbor': upscaled_nn,
    'bicubic': upscaled_bicubic,
    'lanczos': upscaled_lanczos,
    'ai': upscaled_ai
}

for name, img in images.items():
    cropped = img.crop(box=(280, 145, 395, 260))
    cropped.save(f'../figures/{name}_cropped.png')
