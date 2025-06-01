import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import threading
import time
import collections
import torch
from video_eval_encapsulations import ViolenceDetector

class ViolenceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepViolence Detector")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")
        
        # Model and video state
        self.detector = None
        self.cap = None
        self.is_playing = False
        self.frame_buffer = collections.deque(maxlen=32)
        self.prediction_interval = 5  # Predict every 5 frames
        
        self.current_prediction = {
        "prediction": "WAITING",
        "confidence": 0,
        "color": "gray"  # Default neutral color
        }
        
        # GUI styling
        self.setup_styles()
        self.create_widgets()
        self.load_model()
        
        # Start periodic GUI updater
        self.update_gui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background='#2c3e50')
        style.configure('TLabel', background='#2c3e50', foreground='#ecf0f1')
        style.configure('TButton', font=('Helvetica', 10, 'bold'), padding=5)
        style.configure('Title.TLabel', font=('Helvetica', 20, 'bold'))
        style.configure('Prediction.TLabel', font=('Helvetica', 18, 'bold'))
        style.configure('Red.TLabel', foreground='#e74c3c')
        style.configure('Green.TLabel', foreground='#2ecc71')
        style.configure('Yellow.TLabel', foreground='#f39c12')
        
        style.map('TButton',
                background=[('active', '#3498db'), ('pressed', '#2980b9')],
                foreground=[('active', 'white')])

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Video display
        video_frame = ttk.LabelFrame(main_frame, text="Live Feed")
        video_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0,20))
        
        # Fixed-size canvas for 800x450 video
        self.video_canvas = tk.Canvas(video_frame, bg="black", bd=0, highlightthickness=2, width=800, height=450)
        self.video_canvas.pack(pady=10)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="w")
        
        ttk.Button(control_frame, text="Open Video", command=self.open_video).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Start Webcam", command=self.start_webcam).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Stop", command=self.stop).pack(fill=tk.X, pady=5)
        
        # Status panel
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.pack(fill=tk.X, pady=10)
        
        # Prediction panel
        prediction_frame = ttk.Frame(main_frame)
        prediction_frame.grid(row=1, column=1, sticky="e")
        
        ttk.Label(prediction_frame, text="VIOLENCE DETECTION", style='Title.TLabel').pack(anchor="w")
        
        self.prediction_text = ttk.Label(prediction_frame, text="WAITING", style='Prediction.TLabel')
        self.prediction_text.pack(anchor="w", pady=(10,0))
        
        self.confidence_meter = ttk.Progressbar(prediction_frame, orient="horizontal", length=300)
        self.confidence_meter.pack(fill=tk.X, pady=5)
        
        self.confidence_text = ttk.Label(prediction_frame, text="Confidence: 0%")
        self.confidence_text.pack(anchor="w")
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)


    def load_model(self):
        try:
            self.status_label.config(text="Status: Loading model...")
            self.root.update()
            
            self.detector = ViolenceDetector(model_path="VD_classification.pt")
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            self.status_label.config(text=f"Status: Model loaded ({device})")
        except Exception as e:
            self.status_label.config(text=f"Status: Model load failed - {str(e)}")
            self.prediction_text.config(text="MODEL ERROR", style='Red.TLabel')

    def open_video(self):
        self.stop()
        filepath = filedialog.askopenfilename(filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*")
        ])
        
        if filepath:
            self.cap = cv2.VideoCapture(filepath)
            self.start_processing()

    def start_webcam(self):
        self.stop()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Status: Webcam not available")
            return
        self.start_processing()

    def start_processing(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        self.is_playing = True
        self.frame_buffer.clear()
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
        
        # Start prediction thread
        self.prediction_thread = threading.Thread(target=self.process_predictions, daemon=True)
        self.prediction_thread.start()
        
        self.status_label.config(text="Status: Processing...")

    def process_video(self):
        frame_count = 0
        while self.is_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Store every frame (no skipping)
            self.frame_buffer.append(frame)
            frame_count += 1
            
            # Display every frame (for smooth video)
            self.display_frame(frame)
            
            # Slightly faster capture to fill buffer quicker
            # time.sleep(0.015)  # ~60 FPS capture

    def process_predictions(self):
        while self.is_playing:
            if len(self.frame_buffer) >= 16:  # Wait until we have enough frames
                try:
                    # Get last 16 frames (for SWIN3D temporal analysis)
                    frames = list(self.frame_buffer)[-16:]
                    
                    # Convert all frames to RGB and preprocess
                    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
                    
                    # Get prediction for the sequence
                    result = self.detector.predict_frame_sequence(rgb_frames)
                    
                    # Update current prediction
                    self.current_prediction = {
                        "prediction": "VIOLENT" if result["prediction"] == "Violent" else "SAFE",
                        "confidence": int((result["confidence"] if result["prediction"] == "Violent" else 1 - result["confidence"]) * 100),
                        "color": "red" if result["prediction"] == "Violent" else "green"
                    }
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    self.current_prediction = {
                        "prediction": "ERROR",
                        "confidence": 0,
                        "color": "yellow"
                    }
            
            time.sleep(0.3)  # Predict at 30 FPS

    def display_frame(self, frame):
        # Resize and convert for display
        display_frame = cv2.resize(frame, (800, 450))
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.video_canvas.imgtk = imgtk
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        
        # Update border based on prediction
        border_color = self.current_prediction.get("color", "black")
        self.video_canvas.config(highlightbackground=border_color)

    def update_gui(self):
        pred = self.current_prediction
        
        # Safely get values with defaults
        prediction_text = pred.get("prediction", "WAITING")
        confidence_value = pred.get("confidence", 0)
        color = pred.get("color", "gray")
        
        # Update prediction display
        style_name = f"{color.title()}.TLabel"
        self.prediction_text.config(
            text=prediction_text,
            style=style_name
        )
        
        # Update confidence display
        self.confidence_meter["value"] = confidence_value
        self.confidence_text.config(text=f"Confidence: {confidence_value}%")
        
        # Update video border
        self.video_canvas.config(highlightbackground=color)
        
        # Schedule next update
        self.root.after(100, self.update_gui)

    def stop(self):
        self.is_playing = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.current_prediction = {"prediction": "STOPPED", "confidence": 0}
        self.video_canvas.delete("all")
        self.video_canvas.config(highlightbackground="black")
        self.status_label.config(text="Status: Stopped")

    def on_closing(self):
        self.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ViolenceDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()