from collections import OrderedDict
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import requests
import os

nltk.download("punkt")


class Text2Tags:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "fabiochiu/t5-base-tag-generation"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "fabiochiu/t5-base-tag-generation", torch_dtype=torch.float16
        )
        self.model.to(device)

    def __call__(self, text: str):
        inputs = self.tokenizer(
            [text], max_length=512, truncation=True, return_tensors="pt"
        )
        inputs = inputs.to(device=self.device)
        output = self.model.generate(
            **inputs, num_beams=8, do_sample=True, min_length=10, max_length=64
        )
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[
            0
        ]
        tags = set(decoded_output.strip().split(", "))
        return list(tags)

    def to(self, device):
        self.model.to(device)


class TitleGeneration:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "fabiochiu/t5-small-medium-title-generation"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "fabiochiu/t5-small-medium-title-generation",
            torch_dtype=torch.float16,
        )
        self.model.to(device)

    def __call__(self, text: str):
        inputs = self.tokenizer(
            [text], max_length=512, truncation=True, return_tensors="pt"
        )
        inputs = inputs.to(device=self.device)
        output = self.model.generate(
            **inputs, num_beams=8, do_sample=True, min_length=10, max_length=64
        )
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[
            0
        ]
        tags = set(decoded_output.strip().split(", "))
        return list(tags)

    def to(self, device):
        self.model.to(device)


class WeatherAPI:
    def __init__(self, device):
        self.device = device
        self.key = os.environ.get("WEATHER_API_KEY", "")
        self.url_api_weather = (
            "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{{location}}?&unitGroup=metric&key="
            + self.key
        )

    def get(self, city):
        city = city.replace(" ", "%20")
        url = self.url_api_weather.replace("{{location}}", city)
        print(f"url: {url}")
        return requests.get(url).json()

    def remove(self, item):
        item.pop("hours", None)
        item.pop("source", None)
        item.pop("stations", None)
        item.pop("icon", None)
        item.pop("windgust", None)
        item.pop("moonphase", None)
        item.pop("datetimeEpoch", None)
        item.pop("sunriseEpoch", None)
        item.pop("sunsetEpoch", None)
        item.pop("solarenergy", None)
        item.pop("feelslike", None)
        item.pop("feelslikemin", None)
        item.pop("feelslikemax", None)
        item.pop("precip", None)
        return item

    def __call__(self, loc: str) -> dict:
        result = self.get(loc)
        json_data = OrderedDict()
        json_data["latitude"] = result["latitude"]
        json_data["longitude"] = result["longitude"]
        json_data["resolvedAddress"] = result["resolvedAddress"]
        json_data["address"] = result["address"]
        json_data["timezone"] = result["timezone"]
        json_data["tzoffset"] = result["tzoffset"]
        json_data["description"] = result["description"]
        json_data["measurement_units"] = [
            {"Variable": "Temperature, Heat Index & Wind Chill", "Units": "Celsius"},
            {"Variable": "Precipitation", "Units": "Millimeters"},
            {"Variable": "Snow", "Units": "Centimeters"},
            {"Variable": "Wind & Wind Gust", "Units": "Kilometers Per Hour"},
            {"Variable": "Visibility", "Units": "Kilometers"},
            {"Variable": "Pressure", "Units": "Millibars (Hectopascals)"},
            {"Variable": "Solar Radiation", "Units": "W/m^2"},
        ]
        json_data["days"] = []
        result.pop("alerts")
        # json_data.pop("stations")
        result["days"] = result["days"][::3]
        for item in result["days"]:
            json_data["days"].append(self.remove(item))

        json_data["currentConditions"] = self.remove(result["currentConditions"])
        json_data["currentConditions"]["datetime"] = (
            json_data["days"][0]["datetime"]
            + " "
            + json_data["currentConditions"]["datetime"]
        )
        print(json_data)
        return json_data

    def to(self, device):
        pass


if __name__ == "__main__":
    # text = """
    # Python is a high-level, interpreted, general-purpose programming language. Its
    # design philosophy emphasizes code readability with the use of significant
    # indentation. Python is dynamically-typed and garbage-collected.
    # """
    text = """A group of people are walking around in a field and a dog is walking in front of them and then a woman is walking in front of them and then a man is walking in front of them and then a dog is walking in front of them and then a woman is walking in front of them and then a man is walking in front of them and then a woman is walking in front of them and then a man is walking in front of them and then a woman is walking in front of them and then a man is walking in front of them.
    """
    model = Text2Tags("cuda:0")
    print(model(text))
