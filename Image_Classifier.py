
# import streamlit as st
# import base64
# st.markdown('<h1 style="color:black;">Vgg 19 Image classification model</h1>'   , unsafe_allow_html=True)
# st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
# st.markdown('<h3 style="color:gray;"> street,  buildings, forest, sea, mountain, glacier</h3>', unsafe_allow_html=True)

# # # background image to streamlit

# # @st.cache(allow_output_mutation=True)
# # def get_base64_of_bin_file(bin_file):
# #     with open(bin_file, 'rb') as f:
# #         data = f.read()
# #     return base64.b64encode(data).decode()

# # def set_png_as_page_bg(png_file):
# #     bin_str = get_base64_of_bin_file(png_file) 
# #     page_bg_img = '''
# #     <style>
# #     .stApp {
# #     background-image: url("data:image/png;base64,%s");
# #     background-size: cover;
# #     background-repeat: no-repeat;
# #     background-attachment: scroll; # doesn't work
# #     }
# #     </style>
# #     ''' % bin_str
    
# #     st.markdown(page_bg_img, unsafe_allow_html=True)
# #     return

# # set_png_as_page_bg('/content/background.webp')

# upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
# c1, c2= st.columns(2)
# if upload is not None:
#   im= Image.open(upload)
#   img= np.asarray(im)
#   image= cv2.resize(img,(224, 224))
#   img= preprocess_input(image)
#   img= np.expand_dims(img, 0)
#   c1.header('Input Image')
#   c1.image(im)
#   c1.write(img.shape)






import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

st.title("Image Classifier")

# You can use any pre-trained model from Keras Applications

from keras.applications.mobilenet_v2 import MobileNetV2
model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:8501/')

#model = tf.keras.models.load_model('path/to/your/model.h5')

#@st.cache(allow_output_mutation=True)
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(image)#[0]

    # Process your result for human
    pred_proba = "{:.3f}".format(np.amax(prediction))    # Max probability
    pred_class = decode_predictions(prediction, top=1)   # ImageNet Decode

    result = str(pred_class[0][0][1])               # Convert to string
    result = result.replace('_', ' ').capitalize()
      
    # Serialize the result, you can add additional fields
    #return jsonify(result=result, probability=pred_proba)

    return result, pred_proba
    # classes = ['class1', 'class2', 'class3'] # Replace with your own class names
    # results = {}
    # for i in range(len(classes)):
    #     results[classes[i]] = round(prediction[i]*100, 2)
    # return results

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    results, probability = predict(image)
    probability=float(probability)
    #st.write(probability)
    st.markdown(f'<p style="color:#39FF14;font-size:30px;font-weight:bold;border-radius:2%;text-align: center;">{results}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:white;font-size:20px;border-radius:2%;text-align: center;">{probability}</p>', unsafe_allow_html=True)
    # for key in results:
    #     st.write(f"{key}: {results[key]}%")