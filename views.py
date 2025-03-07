from user.models import user_profile
from django.shortcuts import render, redirect
import cv2
from django.core.mail import EmailMultiAlternatives
from PIL import Image
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import time
from datetime import datetime
import pythoncom
from win32com.client import constants, Dispatch
from translate import Translator
import random
from gtts import gTTS
from playsound import playsound
import joblib

# from sklearn.externals import joblib
# filename = "Z:\BTP\knk\static\gest2aud\HOG_full_newaug.sav"
import keras

import pickle
import h5py

from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import os
import json
from django.http import HttpResponse
from keras.models import load_model
import skimage
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import numpy as np
import cv2
import keras
import tensorflow as tf
from string import ascii_uppercase
from language_tool_python import LanguageTool


parser = LanguageTool('en-US')


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)


model = keras.models.load_model(
   'C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/gest2aud/CNN_model.h5'
    )

alpha_dict = {
    0: '1',  1: '2',  2: '3',  3: '4',  4: '5', 
    5: '6',  6: '7',  7: '8',  8: '9',
    9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E',
    14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J',
    19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O',
    24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T',
    29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y',
    34: 'Z'
}


def preprocess_image(image):

    
    # Converting image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Converting image to HSV format
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Defining boundary levels for skin color in HSV
    skin_color_lower = np.array([0, 40, 30], np.uint8)
    skin_color_upper = np.array([43, 255, 255], np.uint8)

    # Producing mask for skin color
    skin_mask = cv2.inRange(hsv_img, skin_color_lower, skin_color_upper)

    # Removing noise from the mask
    skin_mask = cv2.medianBlur(skin_mask, 5)

    # Applying Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Extracting hand by applying mask
    hand = cv2.bitwise_and(gray_img, gray_img, mask=skin_mask)

    # Get edges by Canny edge detection
    canny = cv2.Canny(hand, 60, 60)

    return canny


def test_image(image):
    pred = model.predict(image)
    letter = alpha_dict[np.argmax(pred)]
    if letter == "None":
        return " "
    return letter



word1 = ""

def convert(gestures):
    word = ""
    print("Convert multiple is called")

    for image in gestures:
        print("image is called")
        print(
            "---------------------------------------------------------------------Next gesture-----------------------------------------------------------")
        temp_word = test_image(image)
        print("word: " + word)
        word += temp_word
    

    word = word.lower()
    print(word)
    
    try:
        matches = parser.check(word)
        corrected_words = [match.corrected for match in matches]
        final_word = " ".join(corrected_words)
    except Exception as e:
        print(f"Error checking text: {e}")
        final_word = word  # Use original word if an error occurs
    
    return final_word


def convert_single(gestures):
    global word1
    print("Cnvert single is called")

    for image in gestures:
        print("image is called")
        print(
            "---------------------------------------------------------------------Next gesture-----------------------------------------------------------")
        temp_word = test_image(image)
        print("word: " + word1)
        if temp_word == " ":
            speaker = Dispatch("SAPI.SpVoice")  # Create SAPI SpVoice Object
            speaker.Speak(word1)  # Process TTS
            del speaker
            word1 = ""
        word1 += temp_word

    

@csrf_exempt
def take_snaps(request):
    if not request.user.is_authenticated:
        return redirect('../login')

    cam = cv2.VideoCapture(0)  # Initialize the webcam
    gestures = []  # List to store preprocessed images
    img_counter = 0
    x1 = datetime.now()  # Start timing for intervals
    initial = 0

    while True:
        # time.sleep(5)
        x2 = datetime.now()
        #time.sleep(5)
        ret, frame = cam.read()
        if not ret:
            print("Error: Couldn't access webcam")
            break

        frame = cv2.flip(frame, 1)  # Mirror effect
        cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)  # Outline ROI
        frame_crop = frame[10:300, 320:620]  # Extract ROI

        # Preprocess the image
        preprocessed_image = preprocess_image(frame_crop)

        # Show the preprocessed image in a separate OpenCV window
        cv2.imshow("Preprocessed Frame", preprocessed_image)  # New window to display preprocessed image

        # Resize and reshape the image for the model
        reshaped_image = np.expand_dims(cv2.resize(preprocessed_image, (100, 100)), axis=0)
        reshaped_image = reshaped_image[..., np.newaxis]

        # Predict with the model
        prediction = model.predict(reshaped_image)
        predicted_class = np.argmax(prediction)

        # Convert the predicted class to the corresponding letter
        predicted_letter = alpha_dict.get(predicted_class, "Unknown")  # Handle invalid index

        # Display the predicted letter on the original frame
        cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.imshow("Main Frame", frame)  # Main window to display the original frame


        # If 8 seconds have passed, capture the gesture
        if (x2 - x1).seconds >= 10:
            x1 = x2
            initial += 1
            if initial > 1:
                gestures.append(reshaped_image)  # Store the preprocessed image
                img_counter += 1

        k = cv2.waitKey(1)  # Listen for key press
        if k == 27:  # Exit on ESC key
            break

    cam.release()  # Release webcam
    cv2.destroyAllWindows()  # Close OpenCV windows

    print("Number of images captured:", img_counter)

    # Convert gestures to a word using a function like 'convert'
    max_word = convert(gestures)  # Assuming 'convert' converts the gestures to text
    
    # Text-to-speech to speak the converted word
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(max_word)  # TTS
    del speaker

    # Return the converted word as JSON
    data = {'max_word': max_word}
    json_data = json.dumps(data)
    return HttpResponse(json_data, content_type="application/json")



def gest_keyboard(request):
    if request.user.is_authenticated:
        context = {}
        if request.method == "POST":
            print(request.POST['gest_text'])
            gest_text = request.POST['gest_text']
            pythoncom.CoInitialize()

            speaker = Dispatch("SAPI.SpVoice")  # Create SAPI SpVoice Object
            speaker.Speak(gest_text)  # Process TTS
            del speaker

            context = {'gest_text': gest_text}
            print("ddd")
        return render(request, 'gest2aud/gest_keyboard.html', context)
    else:
        return redirect('../login')

def emergency(request):
    if (request.method == "POST"):
        print(request.POST)
        # print(request.user)
        print(user_profile.objects.get(user=request.user))
        usr = user_profile.objects.get(user=request.user)
        mail_text = []
        # print(request.POST['csrfmiddlewaretoken'])
        for i in request.POST:
            if (i != "csrfmiddlewaretoken"):
                mail_text.append(request.POST[i])
        print(mail_text)

        EMAIL = []
        EMAIL.append(usr.Email1)
        EMAIL.append(usr.Email2)
        EMAIL.append(usr.Email3)
        EMAIL.append(usr.Email4)
        EMAIL.append(usr.Email5)
        print(EMAIL, "-------------------------------------------------")
        for i in EMAIL:
            subject, from_email, to = "Emergency Message", "tempuxyz@gmail.com", i
            text_content = "This is an emergnecy message from your deaf friend"
            text_content += '\n'
            for i in mail_text:
                text_content += i
                text_content += '\n'

            msg = EmailMultiAlternatives(
                subject, text_content, from_email, [to])
            msg.send()

    return render(request, 'gest2aud/Emergency.html', {})
    

@csrf_exempt
def language_convert(request):
    if request.user.is_authenticated:
        print("were hereeeeeeeeee babayyyy")
        body_unicode = request.body.decode('utf-8') 	
        body1 = json.loads(body_unicode)
        print(body1)
        
        print(body1["lngCode"])
        translator= Translator(to_lang=body1["lngCode"])
        translation = translator.translate(body1["text"])
        print(translation)

        myobj = gTTS(text=translation, lang=body1["lngCode"], slow=False)

        # Saving the converted audio in a mp3 file named
        # welcome
        myobj.save("C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/gest2aud/welcome.mp3")
        time.sleep(2)
        # Playing the converted file
        playsound("C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/gest2aud/welcome.mp3")
        data = {}
        data['max_word'] = translation
        json_data = json.dumps(data)
        os.remove("C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/gest2aud/welcome.mp3")
        return HttpResponse(json_data, content_type="application/json")
    else:
         return redirect('../login')
