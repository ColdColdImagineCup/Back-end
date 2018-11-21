import os
import sys

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

PREDICTION_KEY = "c6b2b727acbd433baadd201c300e879f"

ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com"

PROJECT_ID = "13e8cd17-489a-4d82-a289-f285c233fff8"

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")

def predict_project(prediction_key,file_path):
    predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

    with open(os.path.join(IMAGES_FOLDER,file_path), mode="rb") as test_data:
        results = predictor.predict_image(PROJECT_ID, test_data.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))


if __name__ == "__main__":

    predict_project(PREDICTION_KEY,"Test/test_image.jpg")