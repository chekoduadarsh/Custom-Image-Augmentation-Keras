# image-augmentation-keras
This is an Image Augmentation library designed to suppot `ImageDataGenerator()` from `keras.preprocessing` library.

### Welcoming contribution to this repo!!! you can include any image processing techiques to this code 

Presently this library contains following methods,
1. contrast_stretch
2. histogram_equalization
3. CLAHE

## Implimentation
```python3
from image_augmentation_keras import Image_Augmentation

CLAHE = Image_Augmentation(method="CLAHE")
 
datagen = ImageDataGenerator(
        preprocessing_function=CLAHE())
 
```

