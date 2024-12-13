import random
# select a specific batch
# images, labels = next(iter(test_data))

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import openai
import os
from PIL import Image
import base64
import io
from openai import OpenAI
import chainlit as cl
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import numpy as np
from datasets.download_data import dataset, class_names



# Load the trained model
@cl.on_chat_start
async def start():
    # cl.user_session.set("model", load_model("plant_disease_model.h5"))
    
    await cl.Message(content = "Please upload Image with text").send()



# Assuming the model is preloaded
model = load_model("plant_disease_model.h5")


import chainlit as cl
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model("plant_disease_model.h5")

# @cl.on_message
async def pred_disease(images):
    """
    Handles a message to predict plant disease from an uploaded image and displays the image.
    """
    try:
       
        # Preprocess the image for prediction
        img = keras_image.load_img(images, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image

        # Make predictions
        predictions = model.predict(img_array)
        print(f"Predictions: {predictions}") 

        # Ensure the predictions array is not empty and has the expected shape
        if predictions.shape[0] == 0:
            raise ValueError("Prediction returned an empty array.")

        predicted_class = np.argmax(predictions, axis=1)
        print(f"Predicted class index: {predicted_class}")  # Debugging predicted class index

        # Define class names correctly according to the model's output
        
        class_names_list = class_names()
        predicted_label = class_names_list[predicted_class[0]]

       

        # Send the result back to the user
        # await cl.Message(content=f"Predicted Disease: {predicted_label}").send()
        return predicted_label

    except Exception as e:
        # Handle errors and send an appropriate message to the user
        print(f"An error occurred during prediction: {str(e)}")


api = "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"

# API_KEY = os.getenv("OPENAI_API_KEY")

def get_chatgpt_diagnosis(disease):

    client = OpenAI(
        api_key=API_KEY,  # This is the default and can be omitted
    )

    chat_completion = client.chat.completions.create(
       messages=[
                    {
                        "role": "system",
                        "content": "You are an agricultural expert specializing in plant disease treatment. "
                                   "Provide comprehensive, practical treatment recommendations."
                                   
                    },
                    {
                        "role": "user",
                        "content": f"Based on this plant disease analysis,please show plant name and Disease name first then  provide detailed treatment recommendations : {disease}"
                    }
                ],
        model="gpt-4o",
    )
    return chat_completion





@cl.on_message
async def op(msg: cl.Message):
    # Check if the message contains any files (image input)
    images = [file for file in msg.elements if "image" in file.mime]
    
    if images:  # If images are attached
        img_path = images[0].path
        
        # Display the uploaded image
        image = cl.Image(path=img_path, name="uploaded_image", display="inline")
        await cl.Message(
            content="Here is the uploaded image:",
            elements=[image]  # Attach the image to the message
        ).send()
        
        # Get the predicted disease from the image
        predicted_disease = await pred_disease(img_path)
        
        # Then pass the predicted disease to get_chatgpt_diagnosis
        response = get_chatgpt_diagnosis(predicted_disease)
        
        # Send the response from ChatGPT
        await cl.Message(
            content=f"{response.choices[0].message.content}", 
            author="plantcure"
        ).send()
    
    elif msg.content:  # If only text is provided
        # Directly pass the text to ChatGPT for diagnosis
        response = get_chatgpt_diagnosis(msg.content)
        
        # Send the response from ChatGPT
        await cl.Message(
            content=f"{response.choices[0].message.content}", 
            author="plantcure"
        ).send()
    
    else:
        # If no valid input is provided
        await cl.Message(
            content="Please provide an image or text for processing.",
            author="plantcure"
        ).send()










# @cl.on_message
# async def op(msg: cl.Message):
#     img_path = None
    
#     # Check if the message contains an image
#     # if msg.content:
#     images = [file for file in msg.elements if "image" in file.mime] # Get the image path from the attachment
#     # else:
#     #     # If no image is attached, ask the user to upload one
#     #     images = await cl.AskFileMessage(
#     #         content="Please upload an image of the plant to detect the disease.",
#     #         accept=["image/*"]
#     #     ).send()
    
#     # Take the first uploaded image
#     img_path = images[0].path   
        

#     # Display the uploaded image
    
#     if not msg.content:
#         image = cl.Image(path=img_path, name="uploaded_image", display="inline")
#         await cl.Message(
#             content="Here is the uploaded image:",
#             elements=[image]  # Attach the image to the message
#         ).send()
        
#         # Get the predicted disease from the image
#     predicted_disease = await pred_disease(img_path)

#         # Then pass the predicted disease to get_chatgpt_diagnosis
#     response = get_chatgpt_diagnosis(predicted_disease)

#         # Send the response from ChatGPT
#     await cl.Message(content=f"{response.choices[0].message.content}").send()



