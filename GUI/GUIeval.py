import sys
from pathlib import Path
import customtkinter as ctk
from CNN import CNNdataloader, CNNmodel, CNNeval
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))


class eval_frame(ctk.CTkFrame):
      def __init__(self, master):
            super().__init__(master)
            self.master = master

            # Configure the grid layout
            self.grid_columnconfigure(0, weight=1)
            self.grid_columnconfigure(1, weight=2)
            self.grid_rowconfigure(1, weight=1)

            # --- Top Frame ---
            top_frame = ctk.CTkFrame(self)
            top_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

            self.back_button = ctk.CTkButton(top_frame, text="Back to Menu", font=("Minecraftia", 20), command=lambda: master.show_frame("menu"))
            self.back_button.pack(side="left", padx=10, pady=5)
            
            self.load_button = ctk.CTkButton(top_frame, text="Load Statistics", font=("Minecraftia", 20), command=self.load_statistics)
            self.load_button.pack(side="right", padx=10, pady=5)

            self.left_frame = ctk.CTkFrame(self)
            self.left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
            self.left_frame.grid_columnconfigure(0, weight=1)

            self.accuracy_label = ctk.CTkLabel(self.left_frame, text="Accuracy: N/A", font=("Minecraftia", 18))
            self.accuracy_label.pack(padx=20, pady=10)

            self.f1_score_label_title = ctk.CTkLabel(self.left_frame, text="F1 Scores:", font=("Minecraftia", 18, "bold"))
            self.f1_score_label_title.pack(padx=20, pady=10)
            
            self.f1_score_label = ctk.CTkLabel(self.left_frame, text="N/A", font=("Minecraftia", 16), justify="left")
            self.f1_score_label.pack(padx=20, pady=10)
            self.right_frame = ctk.CTkFrame(self)

            self.right_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
            self.right_frame.grid_columnconfigure(0, weight=1)
            self.right_frame.grid_rowconfigure(1, weight=1)

            self.confusion_matrix_title = ctk.CTkLabel(self.right_frame, text="Confusion Matrix", font=("Minecraftia", 18, "bold"))
            self.confusion_matrix_title.pack(padx=10, pady=10)
            
            self.confusion_matrix_label = ctk.CTkLabel(self.right_frame, text="N/A", font=("Minecraftia", 12), justify="left")
            self.confusion_matrix_label.pack(expand=True, fill="both", padx=10, pady=10)
            
            self.canvas_widget = None
            self.model = None 
            self.test_data = None
            

      def load_statistics(self):
            
            self.load_button.configure(state="disabled", text="Loading...")

            self.update_idletasks()
            try:
                  # Initialize model and dataset 
                  
                  self.model = CNNmodel.init_model("default model")
                  
                  tomato_dataset = CNNdataloader.load_tomato_dataset()
                  _,self.test_data,_,_ = CNNdataloader.create_train_and_test_dataset(64, tomato_dataset)

                  accuracy = CNNeval.check_accuracy_from_dataset(self.model,self.test_data)
                  self.accuracy_label.configure(text=f"Accuracy: {accuracy:.2f}%")

                  f1_scores = CNNeval.get_F1_score(self.model,self.test_data)
                  
                  f1_text = ""
                  for class_name, score in f1_scores.items():
                        f1_text += f"{class_name}: {score:.4f}\n"

                  self.f1_score_label.configure(text=f1_text)

                  confusion_matrix = CNNeval.get_confusion_matrix(self.model, self.test_data)
                  self.plot_confusion_matrix(confusion_matrix.numpy()) # Convert tensor to numpy array for plotting


            except Exception as e:
                  self.accuracy_label.configure(text=f"Error: {e}")
                  self.f1_score_label.configure(text="")
                  self.confusion_matrix_label.configure(text="")

            finally:
                  self.load_button.configure(state="normal", text="Load Statistics")
            
      def plot_confusion_matrix(self, confusion_matrix):
            # Clear the previous plot if it exists to prevent overlap
            if self.canvas_widget:
                  self.canvas_widget.destroy()

            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=CNNdataloader.tomato_class_names, 
                    yticklabels=CNNdataloader.tomato_class_names)
        
            ax.set_title("Confusion Matrix", fontsize=16)
            ax.set_xlabel("Predicted Labels", fontsize=12)
            ax.set_ylabel("True Labels", fontsize=12)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
            canvas.draw()

            self.canvas_widget = canvas.get_tk_widget()
            self.canvas_widget.pack(expand=True, fill="both", padx=10, pady=10)

            self.confusion_matrix_label.pack_forget()
