from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

os.chdir(os.path.dirname(__file__))

model = load_model("brain_mri_model.keras")

img_path = "classifier_dataset/data_raw_dir/Testing/glioma/Te-gl_0166.jpg"
img = image.load_img(img_path, target_size=(128,128))

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = img_array / 255.0

pred = model.predict(img_array)

print("Raw prediction:", pred)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

pred_class = np.argmax(pred, axis=1)[0]
print("Predicted class:", class_names[pred_class])

for i, cls in enumerate(class_names):
    print(f"{cls}: {pred[0][i]:.4f}")