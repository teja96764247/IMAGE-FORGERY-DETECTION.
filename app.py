import streamlit as st
import tensorflow as tf



@st.cache_data
def load_model():
    model=tf.keras.models.load_model('forgery.hdf5')
    return model
model=load_model()

st.write("""
       #IMAGE FORGERY DETECTION WEB APP
         """
)

file=st.file_uploader("Please upload an Image",type=['jpg','png'])

from PIL import Image,ImageOps
import numpy as np

def predict_function(img,model):
    size=(64,64)
    image=ImageOps.fit(img,size,Image.ANTIALIAS)
    img_arr=np.asarray(image)
    img_scaled=img_arr/255
    img_reshape=np.reshape(img_scaled,[1,64,64,3])
    prediction=model.predict(img_reshape)
    output=np.argmax(prediction)
    if(output==0):
        return "The Image is ORIGINAL Image"
    else:
        return "The Image is FORGERY Image"
    

if file is None:
    st.text('Please upload an image file')
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    result=predict_function(image,model)
    st.success(result)