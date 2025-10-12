from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
import io, base64, cv2
import os
from app.utils.preprocess import crop_brain_region

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_classification_model.keras")
model = load_model(MODEL_PATH)
model.build((None, 128, 128, 3))
_ = model(np.zeros((1, 128, 128, 3), dtype=np.float32))
model.summary()
LAST_CONV_LAYER = "conv2d_74"
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

def predict_tumor(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_array, cropped, crop_box = preprocess(img)
    # Prediction
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    predicted_label = class_names[pred_class]

    if predicted_label == "notumor":
        tumor_status = "No Tumor Detected"
    else:
        tumor_status = f"Tumor Detected: {predicted_label.capitalize()}"

    # Generate Grad-CAM heatmap
    heatmap = make_smoothgrad_cam(img_array, model, LAST_CONV_LAYER, n_samples=20, noise_level=0.1)

    original = np.array(img) 
    # overlay = overlay_heatmap_on_original(original, heatmap, crop_box, alpha=0.5)
    
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    # Encode Grad-CAM image as base64
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    def encode_img(img):
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")

    cropped_b64 = encode_img(cropped)
    return {
        "predicted_class": predicted_label,
        "tumor_status": tumor_status,
        "probabilities": {cls: float(pred[0][i]) for i, cls in enumerate(class_names)},
        "gradcam_image": img_base64,
        "cropped_image": cropped_b64
    }

def preprocess(img):
    img_array = np.array(img)

    # Step 1️⃣: Crop brain region
    cropped, crop_box = crop_brain_region(img_array)

    # Step 2️⃣: Convert to array for keras
    img_array = image.img_to_array(cropped)

    # Step 3️⃣: Expand dimension (batch=1)
    img_array = np.expand_dims(img_array, axis=0)

    # Step 4️⃣: Normalize to [0,1]
    img_array = img_array / 255.0

    return img_array, cropped, crop_box
# def preprocess(img_pil):
   
#     img = img_pil.resize((128, 128))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0
#     return img_array

def make_smoothgrad_cam(img_array, model, last_conv_layer_name,
                        pred_index=None, n_samples=20, noise_level=0.1):
    # Save multiple Grad-CAM heatmaps
    heatmaps = []

    for i in range(n_samples):
        # Add noise to the input image
        noisy_img = img_array + np.random.normal(
            loc=0, scale=noise_level, size=img_array.shape
        )
        noisy_img = np.clip(noisy_img, 0, 1)

        # Generate Grad-CAM for the noisy image
        heatmap = make_gradcam_heatmap(noisy_img, model, last_conv_layer_name, pred_index)

        heatmaps.append(heatmap)

    # Average the heatmaps
    smooth_heatmap = np.mean(heatmaps, axis=0)

    # Normalize to [0, 1]
    smooth_heatmap = np.maximum(smooth_heatmap, 0)
    smooth_heatmap = smooth_heatmap / (np.max(smooth_heatmap) + 1e-8)

    return smooth_heatmap

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Ensure the model is built
    if not getattr(model, "built", False):
        model.build((None, 128, 128, 3))
        _ = model(np.zeros((1, 128, 128, 3), dtype=np.float32))

    # Manually rebuild the forward graph
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = inputs
    conv_output = None

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x  

    # Create a new model for gradient computation
    grad_model = tf.keras.models.Model(inputs=inputs, outputs=[conv_output, x])

    # Record gradients with respect to the predicted class
    # Change the last layer's activation to linear to get logits
    last = model.layers[-1]
    orig_act = getattr(last, "activation", None)
    if orig_act is not None:
        last.activation = tf.keras.activations.linear  # 临时关闭 softmax

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)  # predictions 现在是 logits
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # After gradient computation, restore original activation
    if orig_act is not None:
        last.activation = orig_act
    # Compute gradients of the class score wrt conv layer output
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Compute Grad-CAM heatmap
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

def overlay_heatmap_on_original(original_img, heatmap, crop_box, alpha=0.2):
    """
    Map Grad-CAM heatmap (on cropped image) back to full-size original MRI.
    original_img: np.array (H, W, 3)
    heatmap: np.array (128, 128)
    crop_box: (x, y, w, h)
    alpha: blending ratio
    Returns: composite image same size as original
    """
    x, y, w, h = crop_box
    H, W = original_img.shape[:2]

    # 1️⃣ Resize heatmap to cropped area size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # 2️⃣ Normalize and convert to color map
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 3️⃣ Create full-size empty heatmap
    full_heatmap = np.zeros_like(original_img, dtype=np.uint8)
    full_heatmap[y:y+h, x:x+w] = heatmap_color

    # 4️⃣ Blend with original
    overlay = cv2.addWeighted(original_img, 1 - alpha, full_heatmap, alpha, 0)
    return overlay