from flask import Flask, render_template, request, jsonify
import base64
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 47)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the CNN model
model = CNN()

model.load_state_dict(torch.load('static/model.pt'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_image', methods=['POST'])
def get_image():
    print(request.get_json())
    data = request.get_json()
    # Get the image data from the request
    data_url = data['imageData']
    # Remove the data header and decode the base64-encoded data
    img_data = base64.b64decode(data_url.split(',')[1])

    # Save the image to a file
    with open('image.jpg', 'wb') as f:
        f.write(img_data)

    image = Image.open("image.jpg")
    image_np = np.array(image)
    image_np = image_np[:, :, 3]

    # Calculate the coordinates of the top-left corner of the crop
    x = (image_np.shape[0] - 256) // 2
    y = (image_np.shape[1] - 256) // 2

    # Crop the array
    cropped_arr = image_np[x:x + 256, y:y + 256]
    image_np = np.resize(cropped_arr, (28, 28))

    # Convert numpy array to PyTorch tensor
    image_tensor = torch.from_numpy(image_np).float()

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
    print(output)

    predicted_label = torch.argmax(output).item()
    print(predicted_label)

    # Get the score from the request
    score = data['score']
    if predicted_label in {11, 22, 32, 34, 42}:
        score += 10
    print(score)

    # Return a response with the image source and score as JSON
    return jsonify({'imgSrc': data_url, 'score': score})

if __name__ == '__main__':
    app.run(debug=True)
