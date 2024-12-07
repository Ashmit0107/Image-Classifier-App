import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to calculate model accuracy
def calculate_model_accuracy(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy * 100

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.title("📸 Image Classification with MobileNetV2")
    st.write("Upload an image, and we'll classify it using the MobileNetV2 model, trained on the **ImageNet** dataset!")

    # Upload image
    uploaded_file = st.file_uploader("📂 Please choose an image (JPG or PNG)...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Your Uploaded Image")
        
        st.write("🔍 Classifying your image...")

        # Load MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

        # Show top predictions
        st.write("### 🎯 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i+1}. **{label.capitalize()}**: {score * 100:.2f}% confidence")
        
        # MobileNetV2 general accuracy (pretrained on ImageNet)
        st.write("### 📊 Model Accuracy:")
        st.write("MobileNetV2 was trained on the ImageNet dataset and achieves approximately **71.8% Top-1 Accuracy** and **91.0% Top-5 Accuracy**.")

# Function for CIFAR-10 model
def cifar10_classification():
    st.title("🖼️ Image Classification with CIFAR-10")
    st.write("Upload an image, and we'll classify it using a custom model trained on the **CIFAR-10** dataset!")

    # Upload image
    uploaded_file = st.file_uploader("📂 Please choose an image (JPG or PNG)...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Your Uploaded Image")
        
        st.write("🔍 Classifying your image...")

        # Load the custom model
        try:
            model = tf.keras.models.load_model('my_model.h5')
        except Exception as e:
            st.error("🚨 Could not load the custom model. Make sure `my_model.h5` is in the app directory.")
            return

        # CIFAR-10 class names
        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Calculate model accuracy
        st.write("🔄 Calculating model accuracy...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0  # Normalize test data
        
        # Convert y_test to one-hot encoding
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=10)

        model_accuracy = calculate_model_accuracy(model, x_test, y_test_one_hot)

        # Show results
        st.write(f"### 🎯 Predicted Class: **{class_names[predicted_class]}**")
        st.write(f"### 🔥 Confidence: **{confidence * 100:.2f}%**")
        st.write(f"### 📊 Model Accuracy: **{model_accuracy:.2f}%**")

# Main function for navigation
def main():
    st.sidebar.title("🔍 Model Selector")
    st.sidebar.write("Choose the model you'd like to use for image classification:")
    choice = st.sidebar.selectbox("🧠 Select a Model", ["CIFAR-10", "MobileNetV2 (ImageNet)"])

    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

# Entry point
if __name__ == "__main__":
    main()
