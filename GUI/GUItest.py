import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image
from CNN import CNNdataloader, CNNmodel, CNNeval

row_count = 2
column_count = 1

class predict_image_frame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        


        for i in range(row_count):
            self.grid_rowconfigure(i, weight=1)
        for i in range(column_count):
            self.grid_columnconfigure(i, weight=1)


        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew",columnspan = column_count )
        

        self.back_button = ctk.CTkButton(top_frame, text="Back to Menu", font=("Minecraftia", 20),command=lambda: master.show_frame("menu"))
        self.back_button.pack(side="left", padx=10, pady=10)

        self.upload_button = ctk.CTkButton(top_frame, text="Upload Image", font=("Minecraftia", 20),command=self.upload_and_predict)
        self.upload_button.pack(side="right", padx=10, pady=10)
        
  
        self.image_label = ctk.CTkLabel(self, text="Upload an image", font=("Minecraftia", 16))
        self.image_label.grid(row=1, column=0, padx=20, pady=20, sticky="nsew", columnspan = column_count)

      
        self.result_label = ctk.CTkLabel(self, text="", font=("Minecraftia", 20, "bold"))
        self.result_label.grid(row=2, column=0, padx=10, pady=10, sticky="ew")


        self.model = CNNmodel.init_model("default model")

    def upload_and_predict(self):
        image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if not image_path:
            return

     #show imgs idk 
        try:
            pil_image = Image.open(image_path)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(400, 400))
            self.image_label.configure(image=ctk_image, text="")
        except Exception as e:
            self.image_label.configure(image=None, text=f"Error loading image: {e}")
            self.result_label.configure(text="")
            return

       #show predictions
        try:
            predicted_class, confidence = CNNeval.predict_image_file(self.model, image_path)
            result_text = f"Prediction: {predicted_class} ({confidence:.2f}% confidence)"
            self.result_label.configure(text=result_text)
        except Exception as e:
            self.result_label.configure(text=f"Error during prediction: {e}")