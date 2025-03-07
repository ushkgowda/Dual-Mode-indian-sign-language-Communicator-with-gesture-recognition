# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# from imutils.video import VideoStream
# from keras.models import load_model
# import cv2
# import numpy as np

# # vs = VideoStream(src=0).start()
# time.sleep(1.0)
# print("Camera Started...")

# # Load the pre-trained model
# model = load_model('C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/CNN_model.h5')


# def preprocess_image():
#     # Reading image

    
#         frame = 'C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/ResizerImage128X1288.jpg'

#         time.sleep(0.1)
#         #cv2.imshow("BW", frame)
#         #cv2.waitKey(3)
#         # Converting image to grayscale
#         gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Converting image to HSV format
#         hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#         # Defining boundary level for skin color in HSV
#         skin_color_lower = np.array([0, 40, 30], np.uint8)
#         skin_color_upper = np.array([43, 255, 255], np.uint8)

#         # Producing mask for skin color
#         skin_mask = cv2.inRange(hsv_img, skin_color_lower, skin_color_upper)

#         # Removing Noise from mask
#         skin_mask = cv2.medianBlur(skin_mask, 5)
#         skin_mask = cv2.addWeighted(skin_mask, 0.5, skin_mask, 0.5, 0.0)

#         # Applying Morphological operations
#         kernel = np.ones((5, 5), np.uint8)
#         skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

#         # Extracting hand by applying mask
#         hand = cv2.bitwise_and(gray_img, gray_img, mask=skin_mask)

#         # Get edges by Canny edge detection
#         canny = cv2.Canny(hand, 60, 60)

#         #img = cv2.imread(canny, cv2.IMREAD_GRAYSCALE)

#         #cv2.imwrite('C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/gest2aud/read2_frame.png', canny)
#         #img = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY)
#         cv2.imshow("image",canny)
#         time.sleep(3)
#         img = cv2.resize(canny, (100, 100))
       
       
#         img = img.astype('float32') / 255.0

#         img = np.reshape(img, (-1, 100, 100, 1))
        
#         predictions = model.predict(img)

#         predicted_class = np.argmax(predictions)

#         print(f"Predicted class label: {predicted_class}")

#     # Display the preprocessed image
#     # cv2.imshow("TW", canny)

#     # cv2.waitKey(3)


# preprocess_image()


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
import numpy as np

# Function to preprocess a single image and save it as JPG
def preprocess_image(image_path, save_path):
    # Reading image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # Converting image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Converting image to HSV format
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining boundary level for skin color in HSV
    skin_color_lower = np.array([0, 40, 30], np.uint8)
    skin_color_upper = np.array([43, 255, 255], np.uint8)

    # Producing mask for skin color
    skin_mask = cv2.inRange(hsv_img, skin_color_lower, skin_color_upper)

    # Removing Noise from mask
    skin_mask = cv2.medianBlur(skin_mask, 5)
    skin_mask = cv2.addWeighted(skin_mask, 0.5, skin_mask, 0.5, 0.0)

    # Applying Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Extracting hand by applying mask
    hand = cv2.bitwise_and(gray_img, gray_img, mask=skin_mask)

    # Get edges by Canny edge detection
    canny = cv2.Canny(hand, 60, 60)

    # Display the preprocessed image
    plt.imshow(canny, cmap='gray')
    plt.title('Preprocessed Image1')
    plt.axis('off')
    plt.show()

    # Save the preprocessed image as JPG
    save_path_jpg = os.path.splitext(save_path)[0] + '.jpg'  # Ensure JPG extension
    cv2.imwrite(save_path_jpg, canny)

    print(f"Preprocessed image saved at: {save_path_jpg}")

# Example usage
image_path = 'C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/ResizerImage128X1288.jpg'
save_path = 'C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/preprocessed_image8new.jpg'
preprocess_image(image_path, save_path)



# Load the pre-trained model
model = load_model('C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/CNN_model.h5')

# Function to preprocess the image
def preprocessing_image(img_path):
    # Load the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to match the model's expected sizing
    img = cv2.resize(img, (100, 100))
    # Normalize the pixel values
    img = img.astype('float32') / 255.0
    # Reshape the image to match the input shape of the model
    img = np.reshape(img, (-1, 100, 100, 1))
    return img


# Path to the image you want to predict
test_img_path = 'C:/Users/sudeep/Desktop/Indian-Sign-Language-Gesture-Recognition-master/preprocessed_image8new.jpg'

# Preprocess the image
test_img = preprocessing_image(test_img_path)

# Make predictions
predictions = model.predict(test_img)

# Get the predicted class label
predicted_class = np.argmax(predictions)

# Print the predicted class label
print(f"Predicted class label: {predicted_class}")