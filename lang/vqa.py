from openai import OpenAI
import base64
import requests
from config.file_config import API_KEY
import pdb

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_headers_payload(api_key, text_content, image_content):
    
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": "gpt-4o-mini",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"{text_content}"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{image_content}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }

    return headers, payload


def send_request(text_info, image_data):
    
    headers, payload = create_headers_payload(api_key=API_KEY, text_content=text_info, image_content=image_data)

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    return response


def send_request_image_path(text_info, image_path):

    # Getting the base64 string
    base64_image = encode_image(image_path)

    return send_request(text_info, base64_image)


if __name__ == "__main__":
    
    text_info = """ I have a Franka FR3 robot arm as shown. It has the following joints:
               Joint1 (Base Joint): Rotates the entire arm around the base,
               Joint2 (Shoulder Joint): Moves the arm up and down (pitch),
               Joint3 (Upper Arm Joint): Rotates the upper arm around its vertical axis (roll), 
               Joint4 (Elbow Joint): Bends the arm, moving it up and down (pitch),
               Joint5 (Forearm Joint): Rotates the forearm around its vertical axis (roll), 
               Joint6 (Wrist Joint): Adjusts the wrists orientation (roll and pitch),
               Joint7 (End Effector Joint): Final rotational adjustment of the end effector (roll),
               Gripper: Grip or release. Tell me the next action you would perform to grasp the object. Just give me one """
    txt_response = send_request_image_path(text_info=text_info, 
                 image_path="/home/nayaka/Desktop/LLMControl/start_img.jpg")
    
    print(txt_response)
