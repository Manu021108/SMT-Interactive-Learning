from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageGrab
import numpy as np
import tensorflow as tf
from tkinter import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import base64
import io
import logging
import base64
from threading import Thread

app = Flask(__name__)

# Flask Routes
@app.route('/')
def home():
    return render_template('home1.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/launch', methods=['GET', 'POST'])
def launch():
    return render_template('launch.html')

@app.post('/identify_shape')
def identify_shape():
    try:
        # Parse JSON body
        data = request.get_json()
        logging.debug("Request JSON data: %s", data)
        
        if 'image' not in data:
            logging.error("No image data provided in the request.")
            return jsonify({"error": "No image data provided"}), 400

        # Extract base64 string
        img_data = data['image']
        logging.debug("Base64 image data: %s", img_data[:30])  # Log first 30 characters for brevity

        # Decode base64
        # print("images: " + img_data)
        img_data = img_data.replace("data:image/png;base64,", "")
        img_bytes = base64.b64decode(img_data)
        logging.debug("Decoded image bytes length: %d", len(img_bytes))

        # Load as PIL image
        # In Python 2.7
        fh = open("input.png", "wb")
        fh.write(img_bytes)
        fh.close()

        # img = Image.open(io.BytesIO(img_bytes))
        # img.save("input.png")  # Save for model prediction
        # logging.info("Image saved as 'input.png'.")

        # # Call the predict function
        result = predict("input.png")
        logging.debug("Prediction result: %s", result)

        return jsonify({"shape": result})  # Return prediction in JSON
    except Exception as e:
        logging.error("Error in identifying shape: %s", e, exc_info=True)
        return jsonify({"error": "Unable to process image"}), 400

# Tkinter GUI Application
class MainApp:
    def __init__(self, master):
        self.master = master
        self.res = ""
        self.pre = [None, None]
        self.bs = 4.5
        self.c = Canvas(self.master, bd=3, relief="ridge", width=300, height=282, bg="white")
        self.c.pack(side=LEFT)

        f1 = Frame(self.master, padx=5, pady=5)
        Label(f1, text="Maths Tutor for Shape", fg="green", font=("", 15, "bold")).pack(pady=10)
        Label(f1, text="Draw a shape to get its formula", fg="green", font=("", 15)).pack()
        Label(f1, text="(Circle, Square, Triangle)", fg="green", font=("", 15)).pack()
        self.pr = Label(f1, text="Prediction: None", fg="blue", font=("", 20, "bold"))
        self.pr.pack(pady=20)

        Button(f1, font=("", 15), fg="white", bg="red", text="Clear Canvas",
               command=self.clear).pack(side=BOTTOM)
        f1.pack(side=RIGHT, fill=Y)

        self.c.bind("<Button-1>", self.putPoint)
        self.c.bind("<ButtonRelease-1>", self.getResult)
        self.c.bind("<B1-Motion>", self.paint)

    def getResult(self, e):
        # Capture the drawing from the canvas
        x = self.master.winfo_rootx() + self.c.winfo_x()
        y = self.master.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1))
        img.save("dist.png")
        
        try:
            self.res = str(predict("dist.png"))  # Get the prediction result
        except Exception as e:
            print("Error during prediction:", e)
            self.res = "Error"
        
        # Display the prediction on the Tkinter interface
        self.pr['text'] = "Prediction: " + self.res

    def clear(self):
        self.c.delete('all')

    def putPoint(self, e):
        self.c.create_oval(e.x - self.bs, e.y - self.bs, e.x + self.bs, e.y + self.bs,
                           outline='black', fill='black')
        self.pre = [e.x, e.y]

    def paint(self, e):
        self.c.create_line(self.pre[0], self.pre[1], e.x, e.y, width=self.bs * 2,
                           fill='black', capstyle=ROUND, smooth=True)
        self.pre = [e.x, e.y]

# Load the trained model
model = load_model("/home/manishji/Smart_Mathematics_Tutor/shapes.h5")

def predict(InputImg):
    try:
        img = image.load_img(InputImg, target_size=(64, 64))  # Load and reshape the image
        x = image.img_to_array(img)  # Convert image to array
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = x / 255.0  # Normalize the image data (if the model expects this)
        pred = model.predict(x)  # Get prediction probabilities
        pred_class = np.argmax(pred, axis=-1)  # Get the index of the highest probability
        index = ['circle', 'square', 'triangle']  # Class labels
        result = str(index[pred_class[0]])  # Get the predicted label
        return result
    except Exception as e:
        print("Error during prediction:", e)
        return "Error"

# Run Flask and Tkinter concurrently
def run_flask():
    app.run(debug=False, use_reloader=False)

if __name__ == "__main__":
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    root = Tk()
    MainApp(root)
    root.title("Smart Mathematics")
    root.resizable(0, 0)
    root.mainloop()
