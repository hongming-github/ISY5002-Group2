import random

def predict_tumor(file_obj) -> bool:
    """
    Dummy tumor classifier.
    Reads the uploaded file (not actually used) and returns random result.
    """
    # Just simulate reading the file
    _ = file_obj.read()
    # Random True/False
    return random.choice([True, False])
