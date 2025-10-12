import cv2

def crop_brain_region(img):
    """
    Crop the brain region by removing black background.
    Works for MRI images with dark background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    gray = cv2.convertScaleAbs(gray)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = img.shape[:2]
        return cv2.resize(img, (128, 128)), (0, 0, w, h)

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = img[y:y+h, x:x+w]
    cropped = cv2.resize(cropped, (128, 128))
    return cropped, (x, y, w, h)