import requests
import pandas as pd
from datetime import datetime

def get_weather():
    API_KEY = "9f3841814a6ba2c0c7729fc1aefda6bc" 
    print("Fetching location...")

    try:
        
        ipinfo = requests.get("https://ipinfo.io").json()
        loc = ipinfo["loc"].split(",")  # 'lat,long'
        latitude, longitude = float(loc[0]), float(loc[1])
        print(" Detected Location:", ipinfo.get("city"), ipinfo.get("region"), ipinfo.get("country"))
    except Exception as e:
        print(" Error getting location:", e)
        return None

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&units=metric&appid={API_KEY}"
        response = requests.get(url).json()

        if "main" in response:
            city = response["name"]
            temperature = response["main"]["temp"]
            weather_desc = response["weather"][0]["description"]
            humidity = response["main"]["humidity"]

            print("\n--- Current Weather ---")
            print("City:", city)
            print("Temperature:", temperature, "°C")
            print("Weather:", weather_desc)
            print("Humidity:", humidity, "%")
        else:
            print("\n API Error:", response)
            return None
        
        print("\n Fetching Current Month Average Rainfall (NASA POWER)...")
        nasa_url = (
            f"https://power.larc.nasa.gov/api/temporal/climatology/point"
            f"?parameters=PRECTOTCORR"
            f"&community=AG"
            f"&longitude={longitude}"
            f"&latitude={latitude}"
            f"&format=JSON"
        )
        nasa_response = requests.get(nasa_url).json()

        rainfall_data = nasa_response.get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})
        if not rainfall_data:
            print(" No rainfall data found")
            return None

        month_abbr = datetime.now().strftime("%b").upper()[:3]
        rainfall_value = rainfall_data.get(month_abbr)

        if rainfall_value is not None:
            monthly_rainfall = round(rainfall_value * 30, 2)
            print(f" Average Rainfall in {month_abbr}: {monthly_rainfall} mm")
        else:
            print(f" Rainfall data not found for {month_abbr}")
            monthly_rainfall = None
        
        return latitude, longitude, temperature, weather_desc, city, humidity, monthly_rainfall

    except Exception as e:
        print("⚠ Error fetching data:", e)
        return None


if __name__ == "__main__":
    result = get_weather()
    if result:
        print("\n Final Data:", result)