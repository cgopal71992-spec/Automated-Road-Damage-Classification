# Import Streamlit
# Streamlit is used to create our web application.
import streamlit as st
import numpy as np
# PIL is used to open and resize images.
from PIL import Image

# Matplotlib is used to create the Grad-CAM overlay.
import matplotlib.pyplot as plt
# Import TensorFlow to load and use our trained deep learning model
import tensorflow as tf

# Import NumPy for image array processing
import numpy as np

# Set the basic page configuration
# This controls the browser tab title and page layout.
st.set_page_config(
    page_title="Road Damage Classification",
    page_icon="🚧",
    layout="centered"
)
# Create a sidebar for project information.
# This gives the application a professional project-style layout.

with st.sidebar:

    st.header("🚧 Project Information")

    st.write("**Automated Road Damage Classification**")

    st.divider()

    st.write("**Deep Learning Model**")
    st.write("Fine-tuned ResNet50")

    st.write("**Damage Classes**")
    st.write("• Pothole")
    st.write("• Crack")
    st.write("• Manhole")

    st.write("**Test Accuracy**")
    st.write("86.75%")

    st.divider()

    st.caption(
        "This application uses deep learning to "
        "classify road damage from uploaded images."
    )
# Path of our saved final ResNet50 model
model_path = r"C:\Users\Gopal\Automated_Road_Damage_Classification\models\road_damage_resnet50_final.keras"


# Load the trained model only once
# This prevents Streamlit from loading the model again
# every time the user interacts with the application.
@st.cache_resource
def load_road_damage_model():

    model = tf.keras.models.load_model(model_path)

    return model


# Load the final trained model
model = load_road_damage_model()
# Access the ResNet50 base model inside our saved model.
# The ResNet50 model contains the convolutional layers
# needed for Grad-CAM visualization.
base_model = model.get_layer("resnet50")


# Select the last convolutional layer of ResNet50.
# This layer contains high-level visual features used
# by the model for classification.
last_conv_layer = base_model.get_layer("conv5_block3_out")
# Create a Grad-CAM heatmap for the predicted class.
# Grad-CAM shows which image regions influenced
# the model's prediction.

def make_gradcam_heatmap(img_array):

    # Create a model that returns:
    # 1. The selected convolutional feature maps
    # 2. The final ResNet50 feature output
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    # Record the gradients during the forward pass.
    with tf.GradientTape() as tape:

        # Get convolutional feature maps and ResNet50 features.
        conv_outputs, features = grad_model(
            img_array,
            training=False
        )

        # Pass ResNet50 features through the same
        # classification layers used by our trained model.
        x = model.get_layer(
            "global_average_pooling2d"
        )(features)

        x = model.get_layer("dense")(x)

        x = model.get_layer(
            "dropout"
        )(x, training=False)

        predictions = model.get_layer(
            "dense_1"
        )(x)

        # Find the predicted class.
        predicted_class = tf.argmax(
            predictions[0]
        )

        # Get the predicted class probability.
        class_probability = predictions[
            :, predicted_class
        ]

    # Calculate gradients of the prediction
    # with respect to the convolutional feature maps.
    grads = tape.gradient(
        class_probability,
        conv_outputs
    )

    # Calculate the average importance
    # of each feature map.
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove the batch dimension.
    conv_outputs = conv_outputs[0]

    # Weight the feature maps using their importance.
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    # Remove the extra dimension.
    heatmap = tf.squeeze(heatmap)

    # Keep only positive values.
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize the heatmap between 0 and 1.
    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    return (
        heatmap.numpy(),
        int(predicted_class),
        float(class_probability[0])
    )




# Main application header
st.title("🚧 Automated Road Damage Classification")

# Short subtitle explaining the purpose of the application
st.caption(
    "AI-powered road damage detection using a fine-tuned ResNet50 model."
)

# Brief instruction for the user
st.write(
    "Upload a road image below to identify potholes, cracks, "
    "or manhole-related damage."
)
# Explain the simple workflow of the application.
st.subheader("🔍 How It Works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("📤 **Upload**")
    st.caption("Upload a road image.")

with col2:
    st.write("🧠 **Analyze**")
    st.caption("ResNet50 analyzes the image.")

with col3:
    st.write("🎯 **Classify**")
    st.caption("Detect the damage type.")

with col4:
    st.write("🛠️ **Recommend**")
    st.caption("Get a suggested action.")

st.divider()

# Create an image upload section
# The user can upload JPG, JPEG, or PNG road images.
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)


# If the user uploads an image, display it.
if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    st.success("Image uploaded successfully!")
    # Convert the uploaded image into a format
    # that TensorFlow can process.
    img = tf.keras.utils.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    # Convert the image into a NumPy array
    img_array = tf.keras.utils.img_to_array(img)

    # Add batch dimension because the model expects
    # input in the format (batch, height, width, channels).
    img_array = np.expand_dims(img_array, axis=0)

    # Apply the same ResNet50 preprocessing
    # that was used during model training.
    img_array = tf.keras.applications.resnet50.preprocess_input(
        img_array
    )

    # Get prediction probabilities from the trained model.
    predictions = model.predict(
        img_array,
        verbose=0
    )

    # Class names must match the order used during training.
    class_names = [
        "pothole",
        "crack",
        "manhole"
    ]

    # Find the class with the highest probability.
    predicted_class = np.argmax(predictions[0])

    # Get the confidence score.
    confidence = predictions[0][predicted_class]

    # Convert class number into class name.
    predicted_label = class_names[predicted_class]
    # Create two columns to display the prediction
    # and class probabilities in a clean layout.
    col1, col2 = st.columns(2)


    # Left column: Final prediction
    with col1:

        st.subheader("🎯 Prediction Result")

        st.success(
            f"Detected Damage: {predicted_label.capitalize()}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # Right column: Probability of each class
    with col2:

        st.subheader("📊 Class Probabilities")

        for i, class_name in enumerate(class_names):

            probability = predictions[0][i]

            st.write(
                f"{class_name.capitalize()}: "
                f"{probability * 100:.2f}%"
            )

            # Display probability using a progress bar.
            st.progress(
                float(probability)
            )

    # Generate Grad-CAM heatmap for the uploaded image.
    # This shows which region influenced the prediction.
    heatmap, gradcam_class, gradcam_confidence = make_gradcam_heatmap(
        img_array
    )

    # Convert the uploaded image into a PIL image.
    # We use the original image size for the overlay.
    original_image = Image.open(uploaded_file).convert("RGB")

    # Resize the Grad-CAM heatmap to match the original image.
    heatmap_resized = Image.fromarray(
        np.uint8(heatmap * 255)
    ).resize(original_image.size)

    # Convert the resized heatmap into a NumPy array.
    heatmap_array = np.array(heatmap_resized)

    # Display the Grad-CAM visualization.
    st.subheader("Grad-CAM Visualization")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Display the original uploaded image.
    ax.imshow(original_image)

    # Overlay the Grad-CAM heatmap.
    ax.imshow(
        heatmap_array,
        cmap="jet",
        alpha=0.45
    )

    ax.set_title(
        f"Model Focus Area - {predicted_label.capitalize()}"
    )

    ax.axis("off")

    # Display the figure in Streamlit.
    st.pyplot(fig)
        # Provide a simple recommendation based on
    # the type of road damage detected by the model.
    recommendations = {
        "pothole": (
            "Inspect and repair the pothole to prevent "
            "further road deterioration and improve road safety."
        ),
        "crack": (
            "Inspect the crack and consider sealing or "
            "repairing it before the damage expands."
        ),
        "manhole": (
            "Inspect the manhole cover and surrounding road "
            "surface for damage, displacement, or safety issues."
        )
    }

    # Get the recommendation for the detected damage type.
    recommendation = recommendations[predicted_label]

    # Display the recommendation to the user.
    st.subheader("Recommended Action")

    st.info(recommendation)
    