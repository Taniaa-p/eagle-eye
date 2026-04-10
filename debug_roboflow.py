import cv2
from inference_sdk import InferenceHTTPClient
import json

img = cv2.imread("/Users/tania/Downloads/WhatsApp Image 2026-04-10 at 10.59.59.jpeg")

try:
    client2 = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="o6ALAmEk1ZcuogAK6cnx"
    )
    result2 = client2.run_workflow(
        workspace_name="tania-p",
        workflow_id="general-segmentation-api",
        images={"image": img},
        parameters={"classes": "License_Plate"}
    )
    print("RAW WORKFLOW RESPONSE:")
    print(json.dumps(result2, indent=2))
except Exception as e:
    print("Workflow model exception:", e)
