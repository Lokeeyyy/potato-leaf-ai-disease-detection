
from django.shortcuts import render
from .models import PredictionHistory
from PIL import Image
import csv
import os
import numpy as np
import tensorflow as tf
import base64

# CNN Model

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "1.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# Disease information
disease_info = {
    "Potato___Early_blight": {
        "name": "Early Blight",
        "description": "A fungal disease that can cause brown spots with rings on potato leaves.",
        "tips": "Remove infected leaves, avoid wetting foliage, and maintain good airflow."
    },
    "Potato___Late_blight": {
        "name": "Late Blight",
        "description": "A serious disease that can cause dark, water-soaked lesions on leaves.",
        "tips": "Remove infected plants safely, avoid overhead watering, and seek local agricultural advice promptly."
    },
    "Potato___healthy": {
        "name": "Healthy",
        "description": "The model classified this image as a healthy potato leaf.",
        "tips": "Continue regular plant monitoring, provide suitable watering, and keep the growing area clean."
    }
}


# HOME PAGE AND DASHBOARD
def home(request):

    # Dashboard counts
    total_predictions = PredictionHistory.objects.count()

    healthy_count = PredictionHistory.objects.filter(
        disease="Healthy"
    ).count()

    early_blight_count = PredictionHistory.objects.filter(
        disease="Early Blight"
    ).count()

    late_blight_count = PredictionHistory.objects.filter(
        disease="Late Blight"
    ).count()

    # Calculate chart percentages
    if total_predictions > 0:
        healthy_percentage = round(
            healthy_count / total_predictions * 100, 1
        )

        early_blight_percentage = round(
            early_blight_count / total_predictions * 100, 1
        )

        late_blight_percentage = round(
            late_blight_count / total_predictions * 100, 1
        )

    else:
        healthy_percentage = 0
        early_blight_percentage = 0
        late_blight_percentage = 0

    # Dashboard context
    context = {
        "total_predictions": total_predictions,
        "healthy_count": healthy_count,
        "early_blight_count": early_blight_count,
        "late_blight_count": late_blight_count,
        "healthy_percentage": healthy_percentage,
        "early_blight_percentage": early_blight_percentage,
        "late_blight_percentage": late_blight_percentage,
    }

    # Image prediction
    if request.method == "POST":

        uploaded_file = request.FILES.get("leaf_image")

        if uploaded_file:

            try:
                uploaded_file.seek(0)

                image_bytes = uploaded_file.read()

                image_base64 = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

                uploaded_file.seek(0)

                image = Image.open(uploaded_file).convert("RGB")

                image = image.resize((256, 256))

                image_array = np.array(image)

                image_array = np.expand_dims(
                    image_array, axis=0
                )

                predictions = model.predict(
                    image_array, verbose=0
                )

                predicted_index = int(
                    np.argmax(predictions[0])
                )

                predicted_class = class_names[predicted_index]

                confidence = float(
                    predictions[0][predicted_index]
                ) * 100

                info = disease_info[predicted_class]

                # Save prediction to SQLite database
                PredictionHistory.objects.create(
                    disease=info["name"],
                    confidence=round(confidence, 2)
                )

                # Prediction result
                context.update({
                    "prediction": info["name"],
                    "description": info["description"],
                    "tips": info["tips"],
                    "confidence": round(confidence, 2),
                    "uploaded_image": image_base64,
                })

                # Refresh dashboard counts after saving
                total_predictions = PredictionHistory.objects.count()

                healthy_count = PredictionHistory.objects.filter(
                    disease="Healthy"
                ).count()

                early_blight_count = PredictionHistory.objects.filter(
                    disease="Early Blight"
                ).count()

                late_blight_count = PredictionHistory.objects.filter(
                    disease="Late Blight"
                ).count()

                context.update({
                    "total_predictions": total_predictions,
                    "healthy_count": healthy_count,
                    "early_blight_count": early_blight_count,
                    "late_blight_count": late_blight_count,
                    "healthy_percentage": round(
                        healthy_count / total_predictions * 100, 1
                    ),
                    "early_blight_percentage": round(
                        early_blight_count / total_predictions * 100, 1
                    ),
                    "late_blight_percentage": round(
                        late_blight_count / total_predictions * 100, 1
                    ),
                })

            except Exception as error:
                print("Prediction error:", error)

                context["error"] = (
                    "Unable to process this image. "
                    "Please upload a valid image and try again."
                )

    return render(
        request,
        "detector/home.html",
        context
    )


# PREDICTION HISTORY PAGE
def history(request):

    records = []

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "oracle_history.csv"
    )

    if os.path.exists(csv_path):

        with open(
            csv_path,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                records.append(row)

    return render(
        request,
        "detector/history.html",
        {"records": records}
    )
