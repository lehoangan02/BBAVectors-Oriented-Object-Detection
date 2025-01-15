import cv2
import threading
import tkinter as tk
from tkinter import Button
from tkinter import scrolledtext

currentImage = 0
image_path = './images/P2806__0.5__0___0.png'
WindowName = "Image Viewer"
stop_event = threading.Event()
root = None

# Function to display the image with OpenCV
def display_image_with_bounding_boxes(image_path):
    print(f"Displaying image: {image_path}")
    
    boxes = read_boxes('./result_dota608/Task1_bridge.txt')  # Read boxes when button is clicked
    image_boxes = boxes[currentImage]  # Get bounding boxes for the image
    # Display the image with bounding boxes
    for box in [image_boxes]:
        imagepath = './images/' + box['image'] + '.png'
        image = cv2.imread(imagepath)
        points = box['points']
        points = [(int(points[i]), int(points[i+1])) for i in range(0, len(points), 2)]
        for i in range(len(points)):
            cv2.line(image, points[i], points[(i+1) % len(points)], (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow(WindowName, image)
    x = 400
    y = 200
    cv2.moveWindow(WindowName, x, y)    
    # Wait for key press or window close
    key = cv2.waitKey(1)  # Non-blocking check for key press

# Callback functions for buttons
def on_button1_click():
    print("Button 1 clicked")
    global currentImage
    currentImage+=1
    print(currentImage)
    global image_path
    display_image_with_bounding_boxes(image_path)

def on_button2_click():
    print("Button 2 clicked")
    global currentImage
    currentImage-=1
    print(currentImage)
    global image_path
    display_image_with_bounding_boxes(image_path)


def read_boxes(file_path):
    boxes = []
    with open(file_path, 'r') as file:

        for line in file:

            parts = line.strip().split()
            image_id = parts[0]
            threshold = float(parts[1])
            coordinates = list(map(float, parts[2:]))
            boxes.append({'image': image_id, 'threshold': threshold, 'points': coordinates})  
    return boxes
def center_window(root, width=800, height=600):
    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate position x and y coordinates
    x = (screen_width // 2) - (width // 2) + 300
    y = (screen_height // 2) - (height // 2)

    root.geometry(f"{width}x{height}+{x}+{y}")
def display_image():
    global root
    global current_image_index, image_boxes, WindowName
    while not stop_event.is_set():
        if current_image_index < 0 or current_image_index >= len(image_boxes):
            continue

        box = image_boxes[current_image_index]
        image_path = f'./images/{box["image"]}.png'
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue

        points = [(int(box["points"][i]), int(box["points"][i + 1])) for i in range(0, len(box["points"]), 2)]
        for i in range(len(points)):
            cv2.line(image, points[i], points[(i + 1) % len(points)], (0, 255, 0), 2)

        cv2.imshow(WindowName, image)
        cv2.moveWindow(WindowName, 400, 200)

def open_text_viewer(file_path):
    global root
    viewer = tk.Toplevel()  # Create a new window
    viewer.title("Text Viewer")
    viewer.geometry("300x200")
    viewer.geometry("+1100+200")
    # Add a scrollable text area
    text_area = scrolledtext.ScrolledText(viewer, wrap=tk.WORD, font=("Arial", 12))
    text_area.pack(expand=True, fill=tk.BOTH)
    
    # Read and display the content of the file
    try:
        with open(file_path, "r") as file:
            text_area.insert(tk.END, file.read())
    except FileNotFoundError:
        text_area.insert(tk.END, f"Error: File '{file_path}' not found.")
    
    text_area.config(state=tk.DISABLED)  # Make the text area read-only

# Main function with Tkinter GUI
def main():
    global image_boxes, root
    image_boxes = read_boxes('./result_dota608/Task1_bridge.txt')
    root = tk.Tk()
    root.title("Image Viewer with Buttons")
    center_window(root, 400, 200)
    root.focus_force()
    Button(root, text="Next bounding box", command=on_button1_click).place(x=0, y=100, width=200, height=50)
    Button(root, text="Previous bounding box", command=on_button2_click).place(x=200, y=100, width=200, height=50)
    Button(root, text="Show Image", command=lambda: display_image_with_bounding_boxes(image_path)).pack(pady=10)
    Button(root, text="Shortcut", command=lambda: open_text_viewer('./shortcuts.txt')).pack(pady=10)
    root.bind("<Right>", lambda event: on_button1_click())
    root.bind("<Left>", lambda event: on_button2_click())
    root.bind("<Escape>", lambda event: root.destroy())
    threading.Thread(target=cv2.waitKey, daemon=True).start()
    root.mainloop()
if __name__ == "__main__":
    main()
