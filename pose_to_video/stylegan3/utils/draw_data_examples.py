from PIL import Image
import zipfile

# Open the zip file
zip_file = zipfile.ZipFile('frames512.zip')

# Create a new image to contain the first 16 images
width, height = (512, 512)
border_width = 2
new_image = Image.new('RGB', (width * 4 + border_width * 3, height * 4 + border_width * 3), color=(255, 255, 255))

# Extract and paste the first 16 PNG images into the new image
skip_images = 2001
num_images = 0
for file_info in zip_file.infolist():
    if num_images >= 16 * skip_images:
        break

    if file_info.filename.endswith('.png'):
        if num_images % skip_images == 0:
            with zip_file.open(file_info) as file:
                image_index = num_images // skip_images
                img = Image.open(file)
                x = (image_index % 4) * (width + border_width) + border_width
                y = (image_index // 4) * (height + border_width) + border_width
                new_image.paste(img.resize((width, height)), (x, y))
                print(f'Pasted image {image_index} ({num_images} images skipped)')
        num_images += 1

# Save the new image
new_image.save('figures/data_examples.png')
