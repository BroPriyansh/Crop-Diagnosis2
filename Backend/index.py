# import json
# def get_cropName(temperature ,humidity):
#   with open("crop_data.json","r") as file:
#     users = json.load(file)

#   for u in users:
#     if u.get("temperature")==temperature and u.get("humidity")==humidity:
#       return u.get("crop-name")
    
#   return None

import json
import math

def get_cropName(temperature, humidity, rainfall=None):
    with open("crop_data.json", "r") as file:
        crops = json.load(file)

    best_crop = None
    best_distance = float("inf")

    for crop in crops:
        temp = crop.get("temperature")
        hum = crop.get("humidity")
        rain = crop.get("rainfall")
        
        dist = math.sqrt(
            (temperature - temp) ** 2 +
            (humidity - hum) ** 2 +
            ((rainfall - rain) ** 2 if rainfall is not None else 0)
        )

        if dist < best_distance:
            best_distance = dist
            best_crop = crop.get("crop_name")

    return best_crop or "No suitable crop found"