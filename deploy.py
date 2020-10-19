from cv2 import *
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import streamlit as st
from PIL import Image

tf.keras.backend.clear_session()

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model =  tf.keras.models.load_model("model_with_975_early_stopping.h5")

def predict_pneumonia(img):
        print(img.shape)
        IMG_SIZE = 300
        img_path = "New Images/Predict/Pneumonia/1.jpeg"
        kernel = np.ones((5,5), np.uint8)   

        norm=cvtColor(img,COLOR_BGR2GRAY)
        th1= equalizeHist(norm)
        eroded = cv2.erode(th1, kernel)
        dilate = cv2.dilate(eroded,kernel)
        eroded2 = cv2.erode(dilate,kernel)
        eroded2 = resize(eroded2,(IMG_SIZE,IMG_SIZE))
        imwrite(img_path,eroded2)

        VAL_DIR = "New Images/Predict"
        val_datagen = ImageDataGenerator(rescale = 1/255)
        val_generator = val_datagen.flow_from_directory(VAL_DIR,target_size=(IMG_SIZE,IMG_SIZE), class_mode='binary', shuffle=False)

        Y_pred = model.predict_classes(val_generator)
        per_list = model.predict(val_generator)

        for i in per_list:
            percentage = i

        for i in Y_pred:
            if i==1:
                image = cv2.putText(img, 'Pneumonia', (30, 30) , cv2.FONT_HERSHEY_SIMPLEX ,
                                1, (255, 0, 0) , 2, cv2.LINE_AA)
            else:
                image = cv2.putText(img, 'Normal', (50, 50) , cv2.FONT_HERSHEY_SIMPLEX ,
                                1, (255, 0, 0) , 2, cv2.LINE_AA)
                                
        return image , percentage
                


def main():
    """Pneumonia Detection App"""

    # st.title("Pneumonia Detection WebApp")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Pneumonia Detection WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        print(image_file)
        image = Image.open(image_file)
        st.text("Original Image")
        st.image(image)

    if st.button("Recognise"):
        image = cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        result_img , percent = predict_pneumonia(image)
        st.image(result_img)
        st.text(percent)


if __name__ == '__main__':
    main()