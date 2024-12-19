import streamlit as st
from roboflow import Roboflow
import requests
from PIL import Image
from io import BytesIO
from PIL import ImageDraw

api_key = "gRihm3PLBZ1dhIcT05Mp"
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("zuby-2-segment")
model = project.version(1).model

def draw_predictions(image, predictions, confidence_threshold=0.25):
    draw = ImageDraw.Draw(image)
    for prediction in predictions:
        if prediction['confidence'] >= confidence_threshold:
            points = prediction['points']
            xy = [(p['x'], p['y']) for p in points]
            fill_color = (255, 255, 0, 153)
            draw.polygon(xy, outline="red", fill=fill_color)  # Draw polygon based on points
    return image


def polygon_area(points):
    """Calculate the area of a polygon given its vertices."""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area

def calculate_grade(area_ratio):
    """Calculate the grade based on the area ratio."""
    if 0 <= area_ratio < 0.1:
        return 1
    elif 0.1 <= area_ratio < 0.3:
        return 2
    elif 0.3 <= area_ratio < 0.5:
        return 3
    elif 0.5 <= area_ratio < 0.7:
        return 4
    elif 0.7 <= area_ratio <= 1.0:
        return 5
    else:
        return None  # In case the area_ratio is out of the expected range


def draw_predictions(image, predictions, confidence_threshold=0.25):
    draw = ImageDraw.Draw(image)
    total_area = 0  # Initialize total area of predictions
    for prediction in predictions:
        if prediction['confidence'] >= confidence_threshold:
            points = prediction['points']
            xy = [(p['x'], p['y']) for p in points]
            fill_color = (255, 255, 0, 153)
            draw.polygon(xy, outline="red", fill=fill_color)  # Draw polygon based on points
            # Calculate and add the area of this polygon to the total area
            total_area += polygon_area(xy)
    return image, total_area


@st.cache(show_spinner=False)
def load_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Function to overlay binary mask on an image
def overlay_mask(image, mask):
    # Convert mask to PIL Image and overlay
    mask_img = Image.open(BytesIO(mask)).convert("RGBA")
    return Image.alpha_composite(image.convert("RGBA"), mask_img)

# Streamlit UI
st.title("SkyLab AG - Plaque Screening")

st.video("https://youtu.be/pRMoyqxiN5Y")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        # Save the uploaded file to a temporary file
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())

        response = model.predict("temp_image.jpg").json()

        try:
            predictions = response['predictions']
            confidence_threshold = 0.25
            
            result_image, total_mask_area = draw_predictions(image.copy(), predictions, confidence_threshold)
            st.image(result_image, caption="Image with Predictions", use_column_width=True)
            # Calculate the ratio
            total_image_area = image.width * image.height
            area_ratio = total_mask_area / (total_image_area*0.4)
            
            st.write(f"Ratio of predicted mask area to the whole image: {area_ratio}")
            print(area_ratio)
            grade = calculate_grade(area_ratio)
            st.write(f"Hygiene Index: {grade}")
            # result_image = draw_predictions(image.copy(), predictions, confidence_threshold)
            # st.image(result_image, caption="Image with Predictions", use_column_width=True)
        except KeyError as e:
            st.error(f"Key error: {e}. Check the response structure.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
