from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import boto3
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse
import os

app = Flask(__name__)

# Configuring AWS credentials
AWS_S3_BUCKET = 'onlinessale-11122024'  # Replace with your S3 bucket name
AWS_REGION = 'ap-south-1'  # Replace with your AWS region
AWS_ACCESS_KEY = 'AKIA5Q4I2WVH4IRCE5NV'  # Replace with your AWS access key
AWS_SECRET_KEY = 'zibuY0rUG+59kf/fLX9Lq2l2wl2AK4gM9HK91wHy'  # Replace with your AWS secret key

# S3 boto for sending the image to s3
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Download image from the URL
def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return response.content  # Return image content as bytes
    else:
        raise ValueError("Failed to download image.")

# Validate the bounding box coordinates
def validate_bounding_box(bbox):
    x_min, y_min, x_max, y_max = bbox
    if x_min >= x_max:
        raise ValueError("Coordinate 'x_max' must be greater than 'x_min'.")
    if y_min >= y_max:
        raise ValueError("Coordinate 'y_max' must be greater than 'y_min'.")
    return True

def remove_background_from_region(image_bytes, x1, y1, x2, y2):
    """
    Removes the background from a specified rectangular region in the image, replacing it with a white background,
    and crops the image to the specified region.
    """
    # Decode the image bytes to a NumPy array
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image.")

    # Ensure coordinates are within the image dimensions
    height, width, _ = image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    # Extract the region of interest (ROI)
    roi = image[y1:y2, x1:x2]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Create a binary mask using Otsu's thresholding
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a 3-channel white background for the ROI
    white_bg = np.full_like(roi, 255)

    # Apply the mask to the ROI to remove the background
    roi_bg_removed = np.where(mask[..., None] == 255, roi, white_bg)

    return roi_bg_removed

def generate_s3_key(image_url):
    """Generates a sanitized S3 object key from the image URL."""
    parsed_url = urlparse(image_url)
    filename = os.path.basename(parsed_url.path)  # Extract the last part of the path
    return f"{filename}.png"

def upload_to_s3(image, filename):
    """Uploads an image to AWS S3 and returns the public URL."""
    # Encode the processed image as PNG
    _, buffer = cv2.imencode('.png', image)

    # Upload image to S3 with public-read access
    s3_client.put_object(
        Bucket=AWS_S3_BUCKET,
        Key=filename,
        Body=buffer.tobytes(),
        ContentType='image/png',
        ACL='public-read'
    )

    # Generate and return the public URL of the uploaded image
    return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{filename}"

@app.route('/remove-background', methods=['POST'])
def remove_background():
    """Endpoint to process the image and remove its background."""
    try:
        # Parse input JSON payload
        data = request.json
        image_url = data.get('image_url')  # Get public image URL
        bbox = data.get('bounding_box')  # Get bounding box coordinates

        # Validate input
        if not image_url or not bbox:
            return jsonify({"error": "Invalid input. Provide image_url and bounding_box."}), 400

        # Validate bounding box
        validate_bounding_box([bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']])

        # Download the image from the provided URL
        image_bytes = download_image(image_url)

        # Process the image: crop and remove background
        processed_image = remove_background_from_region(
            image_bytes, bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
        )

        # Create the filename from the sanitized image URL
        filename = generate_s3_key(image_url)

        # Upload the processed image to S3 and get its public URL
        processed_image_url = upload_to_s3(processed_image, filename)

        # Return success response with URLs
        return jsonify({
            "original_image_url": image_url,
            "processed_image_url": processed_image_url
        }), 200

    except ValueError as e:
        # Handle specific errors (e.g., download failure, invalid bounding box)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": "An unexpected error occurred: " + str(e)}), 500

# Add the new root endpoint
@app.get("/")
def read_root():
    """Root endpoint for a welcome message."""
    return "Welcome to Onlinesales API Assignment!"

if __name__ == '__main__':
    # Run the Flask app on localhost with debugging enabled
    app.run(debug=True, host='0.0.0.0', port=5000)
