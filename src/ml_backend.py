import os

import requests
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size

try:
    # Ignore failure during "label-studio-ml init ml_backend"
    PREDICT_API_URL = os.environ["PREDICT_API_URL"]
    print(PREDICT_API_URL)
except KeyError as e:
    print(e)


class CarDamageSegmentation(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(CarDamageSegmentation, self).__init__(**kwargs)

        print(self.parsed_label_config)

    def reset_model(self):
        pass

    def call_api(self, image_path):
        files = {'file': open(image_path, 'rb')}
        response = requests.post(PREDICT_API_URL, files=files)
        response_json = response.json()
        if response.ok:
            return response_json
        else:
            raise Exception(f'{response.reason}: {response_json}')

    def predict(self, tasks, **kwargs):
        predictions = []
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]
        for task in tasks:
            print(task['annotations'])

            image_link = task['data']['image']
            image_path = '/'+image_link.split('?d=')[1]

            prediction = {
                'result': [
                    {
                        "id": "view_class",
                        "type": "choices",
                        "value": {
                            "choices": [
                                "無法判斷"
                            ]
                        },
                        "to_name": "image",
                        "from_name": "view_class"
                    },
                    {
                        "id": "is_damaged",
                        "type": "choices",
                        "value": {
                            "choices": [
                                "有"
                            ]
                        },
                        "to_name": "image",
                        "from_name": "is_damaged"
                    }
                ]
            }

            api_result = self.call_api(image_path)
            img_width, img_height = get_image_size(image_path)
            results = api_result['results']

            for i, result in enumerate(results):
                rle = result['pred_masks_rle']
                r = {
                    "id": f"brush_{i}",
                    "type": "brushlabels",
                    "value": {
                        "rle": rle,
                        "format": "rle",
                        "brushlabels": [
                            "凹陷"
                        ]
                    },
                    "to_name": "image",
                    "from_name": "damage_class",
                    "image_rotation": 0,
                    "original_width": img_width,
                    "original_height": img_height
                }
                prediction['result'].append(r)

            predictions.append(prediction)

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        pass
