import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageTk
import network  # your updated neural network file
import os
import time

class DigitRecognizerApp:
    def __init__(self, root, neural_net):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.configure(bg="#f5f5f7")
        self.root.minsize(800, 580)
        self.neural_net = neural_net
        
        # Set up styles
        self.setup_styles()
        
        # Create frames
        self.main_frame = ttk.Frame(root, padding="10", style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas frame (left side)
        self.canvas_frame = ttk.Frame(self.main_frame, padding="10", style="CanvasSection.TFrame")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Results frame (right side)
        self.results_frame = ttk.Frame(self.main_frame, padding="10", style="ResultsSection.TFrame")
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Creating title labels
        self.draw_title = ttk.Label(self.canvas_frame, text="Draw a Digit", font=('Helvetica', 18, 'bold'), style="Title.TLabel")
        self.draw_title.pack(pady=(0, 15))
        
        self.results_title = ttk.Label(self.results_frame, text="Recognition Results", font=('Helvetica', 18, 'bold'), style="Title.TLabel")
        self.results_title.pack(pady=(0, 15))
        
        # Canvas for drawing with border
        self.canvas_container = ttk.Frame(self.canvas_frame, style="Canvas.TFrame")
        self.canvas_container.pack(padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_container, width=280, height=280, bg='black', 
                               highlightthickness=0, cursor="pencil")
        self.canvas.pack(padx=2, pady=2)
        
        # Canvas instructions
        self.instructions = ttk.Label(self.canvas_frame, text="Draw a single digit (0-9) in the box above",
                                     wraplength=300, justify="center", style="Instructions.TLabel")
        self.instructions.pack(pady=10)
        
        # Keyboard shortcuts info
        self.shortcuts_label = ttk.Label(self.canvas_frame, text="Shortcuts: Ctrl+R (Recognize), Ctrl+C (Clear)",
                                        wraplength=300, justify="center", style="Shortcuts.TLabel")
        self.shortcuts_label.pack(pady=(0, 10))
        
        # Button frame
        self.button_frame = ttk.Frame(self.canvas_frame, style="ButtonFrame.TFrame")
        self.button_frame.pack(pady=10, fill=tk.X)
        
        # Clear button
        self.clear_btn = ttk.Button(self.button_frame, text="Clear Canvas", command=self.clear_canvas, style="Action.TButton")
        self.clear_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Recognize button
        self.recognize_btn = ttk.Button(self.button_frame, text="Recognize", command=self.recognize_current_digit, style="Action.TButton")
        self.recognize_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Extra buttons frame
        self.extra_button_frame = ttk.Frame(self.canvas_frame, style="ButtonFrame.TFrame")
        self.extra_button_frame.pack(pady=5, fill=tk.X)
        
        # Save drawing button
        self.save_btn = ttk.Button(self.extra_button_frame, text="Save Drawing", command=self.save_drawing, style="Secondary.TButton")
        self.save_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Load drawing button
        self.load_btn = ttk.Button(self.extra_button_frame, text="Load Drawing", command=self.load_drawing, style="Secondary.TButton")
        self.load_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.end_draw)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-r>', lambda event: self.recognize_current_digit())
        self.root.bind('<Control-R>', lambda event: self.recognize_current_digit())
        self.root.bind('<Control-c>', lambda event: self.clear_canvas())
        self.root.bind('<Control-C>', lambda event: self.clear_canvas())
        
        # Previous coordinates for smooth drawing
        self.prev_x = None
        self.prev_y = None
        self.is_drawing = False
        
        # Drawing surface
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        # Processed image display
        self.processed_frame = ttk.LabelFrame(self.results_frame, text="Processed Image (28x28)", style="ProcessedImage.TLabelframe")
        self.processed_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.processed_canvas = tk.Canvas(self.processed_frame, width=140, height=140, bg='black',
                                         highlightthickness=0)
        self.processed_canvas.pack(padx=10, pady=10)
        
        # Recognition result label with styling
        self.result_container = ttk.Frame(self.results_frame, style="Result.TFrame")
        self.result_container.pack(fill=tk.X, padx=10, pady=15)
        
        self.result_label = ttk.Label(self.result_container, text="---", 
                                    font=('Helvetica', 28, 'bold'), justify='center', style="ResultText.TLabel")
        self.result_label.pack(pady=15, padx=10)
        
        # Confidence label
        self.confidence_label = ttk.Label(self.result_container, text="Confidence: ---", 
                                       font=('Helvetica', 16), style="Confidence.TLabel")
        self.confidence_label.pack(pady=(0, 15))
        
        # Confidence visualization
        self.confidence_frame = ttk.LabelFrame(self.results_frame, text="Confidence Levels", style="Confidence.TLabelframe")
        self.confidence_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a frame for the confidence bars
        self.bars_frame = ttk.Frame(self.confidence_frame)
        self.bars_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create confidence bars for each digit
        self.confidence_bars = []
        for i in range(10):
            row_frame = ttk.Frame(self.bars_frame)
            row_frame.pack(fill=tk.X, pady=3)
            
            # Digit label
            digit_label = ttk.Label(row_frame, text=f"Digit {i}:", width=8, style="DigitLabel.TLabel")
            digit_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # Progress bar
            pb = ttk.Progressbar(row_frame, style=f"Confidence{i}.Horizontal.TProgressbar", length=200)
            pb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Value label
            val_label = ttk.Label(row_frame, text="0.0%", width=8, style="PercentLabel.TLabel")
            val_label.pack(side=tk.LEFT, padx=(5, 0))
            
            self.confidence_bars.append((pb, val_label))
        
        # Status bar at the bottom
        self.status_bar = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2), style="StatusBar.TLabel")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize status
        self.update_status("Ready. Draw a digit and click 'Recognize' or press Ctrl+R")

    def setup_styles(self):
        style = ttk.Style()
        
        # Basic colors
        bg_color = "#f5f5f7"
        accent_color = "#4375CD"
        secondary_color = "#e6e6e6"
        
        # Set theme
        style.theme_use('clam')
        
        # Frame styles
        style.configure("Main.TFrame", background=bg_color)
        style.configure("CanvasSection.TFrame", background=bg_color)
        style.configure("ResultsSection.TFrame", background=bg_color)
        style.configure("ButtonFrame.TFrame", background=bg_color)
        
        # Canvas frame styles
        style.configure("Canvas.TFrame", relief=tk.RIDGE, borderwidth=3, background="#333333")
        style.configure("Result.TFrame", relief=tk.GROOVE, borderwidth=2, background="#e9ecef")
        
        # Label styles
        style.configure("Title.TLabel", background=bg_color, foreground="#333333", font=('Helvetica', 18, 'bold'))
        style.configure("Instructions.TLabel", background=bg_color, foreground="#555555", font=('Helvetica', 11))
        style.configure("Shortcuts.TLabel", background=bg_color, foreground="#888888", font=('Helvetica', 10, 'italic'))
        style.configure("ResultText.TLabel", background="#e9ecef", foreground=accent_color, font=('Helvetica', 28, 'bold'))
        style.configure("Confidence.TLabel", background="#e9ecef", foreground="#333333", font=('Helvetica', 16))
        style.configure("DigitLabel.TLabel", background=bg_color, foreground="#333333", font=('Helvetica', 11, 'bold'))
        style.configure("PercentLabel.TLabel", background=bg_color, foreground="#333333", font=('Helvetica', 11))
        style.configure("StatusBar.TLabel", background="#ebebeb", foreground="#555555", font=('Helvetica', 10))
        
        # Button styles
        style.configure("Action.TButton", 
                       font=('Helvetica', 12, 'bold'),
                       background=accent_color, 
                       foreground="white")
        style.map("Action.TButton",
                background=[('active', '#3366bb'), ('pressed', '#2a5298')],
                relief=[('pressed', 'sunken'), ('!pressed', 'raised')])
        
        style.configure("Secondary.TButton", 
                       font=('Helvetica', 11),
                       background=secondary_color)
        style.map("Secondary.TButton",
                background=[('active', '#d9d9d9'), ('pressed', '#cccccc')])
        
        # Labelframe styles
        style.configure("ProcessedImage.TLabelframe", background=bg_color, font=('Helvetica', 11, 'bold'))
        style.configure("ProcessedImage.TLabelframe.Label", background=bg_color, foreground="#333333", font=('Helvetica', 11, 'bold'))
        style.configure("Confidence.TLabelframe", background=bg_color, font=('Helvetica', 11, 'bold'))
        style.configure("Confidence.TLabelframe.Label", background=bg_color, foreground="#333333", font=('Helvetica', 11, 'bold'))
        
        # Progress bar styles for different digits (using different colors)
        colors = [
            "#4285F4",  # 0 - Blue
            "#EA4335",  # 1 - Red
            "#FBBC05",  # 2 - Yellow
            "#34A853",  # 3 - Green
            "#8E44AD",  # 4 - Purple
            "#F39C12",  # 5 - Orange
            "#1ABC9C",  # 6 - Teal
            "#E74C3C",  # 7 - Crimson
            "#3498DB",  # 8 - Light Blue
            "#2C3E50"   # 9 - Dark Blue
        ]
        
        for i, color in enumerate(colors):
            style.configure(f"Confidence{i}.Horizontal.TProgressbar", 
                         background=color, troughcolor="#E0E0E0",
                         bordercolor="#cccccc", lightcolor=color, darkcolor=color)

    def start_draw(self, event):
        self.is_drawing = True
        self.prev_x = event.x
        self.prev_y = event.y
        # Draw a single dot
        self.canvas.create_oval(self.prev_x-10, self.prev_y-10, 
                             self.prev_x+10, self.prev_y+10, 
                             fill="white", outline="white")
        # Draw on the PIL image
        self.draw.ellipse([self.prev_x-10, self.prev_y-10, 
                         self.prev_x+10, self.prev_y+10], 
                         fill="white")
        self.update_status("Drawing...")

    def paint(self, event):
        if not self.is_drawing:
            return
            
        x, y = event.x, event.y
        if self.prev_x and self.prev_y:
            # Draw on canvas
            self.canvas.create_line(self.prev_x, self.prev_y, x, y, 
                                  fill='white', width=20, capstyle=tk.ROUND, joinstyle=tk.ROUND)
            # Draw on image
            self.draw.line([self.prev_x, self.prev_y, x, y], 
                         fill='white', width=20)
        self.prev_x = x
        self.prev_y = y

    def end_draw(self, event):
        self.is_drawing = False
        self.update_status("Ready to recognize (Ctrl+R)")

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.prev_x = None
        self.prev_y = None
        self.result_label.config(text="---")
        self.confidence_label.config(text="Confidence: ---")
        
        # Clear the processed image display
        self.processed_canvas.delete('all')
        
        # Reset progress bars
        for pb, lbl in self.confidence_bars:
            pb['value'] = 0
            lbl.config(text="0.0%")
        
        self.update_status("Canvas cleared")

    def recognize_current_digit(self):
        # Check if anything is drawn
        bbox = self.image.getbbox()
        if bbox is None:
            messagebox.showinfo("Empty Canvas", "Please draw a digit first.")
            return
            
        self.update_status("Recognizing digit...")
        
        # Process recognition with a slight delay to show status
        self.root.after(100, self.do_recognition)

    def do_recognition(self):
        # Preprocess the image
        start_time = time.time()
        img_array = self.preprocess_image()
        
        # Display the processed image
        self.display_processed_image(img_array)
        
        # Reshape for network input (784,1)
        network_input = img_array.reshape(784, 1)
        
        # Get network prediction
        output = self.neural_net.feedforward(network_input)
        prediction = np.argmax(output)
        confidence = float(output[prediction]) * 100
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Update result label
        self.result_label.config(text=f"Digit: {prediction}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Update confidence bars with animation
        self.animate_confidence_bars(output)
        
        # Update status
        self.update_status(f"Recognition complete in {process_time:.3f} seconds")

    def animate_confidence_bars(self, output, step=0):
        if step <= 10:  # 10 animation steps
            # Calculate the current animation progress (0.0 to 1.0)
            progress = step / 10.0
            
            # Update each confidence bar
            for i, (pb, lbl) in enumerate(self.confidence_bars):
                conf_val = float(output[i]) * 100
                # Animate to the target value
                current_val = conf_val * progress
                pb['value'] = current_val
                lbl.config(text=f"{current_val:.1f}%")
            
            # Schedule the next animation step
            self.root.after(30, lambda: self.animate_confidence_bars(output, step + 1))
        else:
            # Final update with exact values
            for i, (pb, lbl) in enumerate(self.confidence_bars):
                conf_val = float(output[i]) * 100
                pb['value'] = conf_val
                lbl.config(text=f"{conf_val:.1f}%")
    
    def preprocess_image(self):
        # Create a copy of the image
        img = self.image.copy()
        
        # Center the digit in the image
        bbox = img.getbbox()
        if bbox is not None:
            # Get bounding box dimensions
            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            
            # Calculate the maximum dimension to maintain aspect ratio
            max_dim = max(width, height)
            
            # Add padding (20% of max dimension)
            padding = int(max_dim * 0.2)
            
            # Create the padded crop area
            crop_left = max(0, left - padding)
            crop_top = max(0, top - padding)
            crop_right = min(img.width, right + padding)
            crop_bottom = min(img.height, bottom + padding)
            
            # Crop to padded bounding box
            img_cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            # Create a new square image with black background
            max_padded_dim = max(crop_right - crop_left, crop_bottom - crop_top)
            img_square = Image.new('L', (max_padded_dim, max_padded_dim), 'black')
            
            # Calculate position to paste (center in square)
            paste_x = (max_padded_dim - (crop_right - crop_left)) // 2
            paste_y = (max_padded_dim - (crop_bottom - crop_top)) // 2
            
            # Paste the cropped image
            img_square.paste(img_cropped, (paste_x, paste_y))
            img = img_square
        
        # Resize to 20x20 (standard MNIST preprocessing)
        img_20 = img.resize((20, 20), Image.Resampling.LANCZOS)
        
        # Create 28x28 image with the digit centered (4px padding on all sides)
        img_28 = Image.new('L', (28, 28), 'black')
        img_28.paste(img_20, (4, 4))
        
        # Convert to numpy array and normalize
        img_array = np.array(img_28).astype(float) / 255.0
        
        # Threshold the image to ensure high contrast
        img_array = (img_array > 0.3).astype(float)
        
        return img_array

    def display_processed_image(self, img_array):
        # Clear previous display
        self.processed_canvas.delete('all')
        
        # Scale up the 28x28 image to fit the 140x140 canvas (5x scaling)
        scale = 5
        img_display = Image.fromarray((img_array * 255).reshape(28, 28).astype(np.uint8))
        img_display = img_display.resize((28*scale, 28*scale), Image.Resampling.NEAREST)
        
        # Convert to PhotoImage for display
        self.processed_photo = ImageTk.PhotoImage(img_display)
        
        # Display on canvas
        self.processed_canvas.create_image(70, 70, image=self.processed_photo)

    def save_drawing(self):
        # Check if anything is drawn
        if self.image.getbbox() is None:
            messagebox.showinfo("Empty Canvas", "There's nothing to save.")
            return
            
        # Get file path from user
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Drawing As"
        )
        
        if not file_path:  # User cancelled
            return
            
        # Save the image
        try:
            self.image.save(file_path)
            self.update_status(f"Drawing saved to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save the drawing: {str(e)}")

    def load_drawing(self):
        # Get file path from user
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
            title="Load Drawing"
        )
        
        if not file_path:  # User cancelled
            return
            
        # Load the image
        try:
            loaded_image = Image.open(file_path)
            
            # Convert to grayscale if needed
            if loaded_image.mode != 'L':
                loaded_image = loaded_image.convert('L')
            
            # Resize to fit canvas if needed
            if loaded_image.size != (280, 280):
                loaded_image = loaded_image.resize((280, 280), Image.Resampling.LANCZOS)
            
            # Update the drawing image
            self.image = loaded_image
            self.draw = ImageDraw.Draw(self.image)
            
            # Clear and redraw canvas
            self.canvas.delete('all')
            img_tk = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(140, 140, image=img_tk)
            self.canvas.image = img_tk  # Keep a reference
            
            self.update_status(f"Loaded drawing from {os.path.basename(file_path)}")
            
            # Automatically recognize the loaded digit
            self.recognize_current_digit()
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load the image: {str(e)}")

    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()  # Force update

def main():
    # Initialize the network
    net = network.Network([784, 256,256,128, 10])
    
    # Load trained weights and biases
    try:
        with open('trained_network.pkl', 'rb') as f:
            weights, biases = pickle.load(f)
            net.weights = weights
            net.biases = biases
        print("Successfully loaded trained network.")
    except Exception as e:
        print(f"Error loading trained network: {e}")
        print("Please train the network first using script.py")
        return
    
    root = tk.Tk()
    app = DigitRecognizerApp(root, net)
    
    # Center the window on the screen
    window_width = 800
    window_height = 580
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()