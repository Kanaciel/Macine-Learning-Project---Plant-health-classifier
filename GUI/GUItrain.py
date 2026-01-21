import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import customtkinter as ctk
from CNN import CNNdataloader, CNNmodel, CNNeval

class training_frame(ctk.CTkFrame):
      def __init__(self, master):
            super().__init__(master)
            self.master = master
        
            for i in range(3):
                  self.grid_rowconfigure(i, weight=1)
            for i in range(1):
                  self.grid_columnconfigure(i, weight=1)

            top_frame = ctk.CTkFrame(self)
            top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

            self.back_button = ctk.CTkButton(top_frame, text="Back to Menu", font=("Minecraftia", 20), command=lambda: master.show_frame("menu"))
            self.back_button.pack(side="left", padx=10, pady=5)

            self.train_button = ctk.CTkButton(top_frame, text="Start Training", font=("Minecraftia", 20), command=self.start_training)
            self.train_button.pack(side="right", padx=10, pady=5)

            self.status_label = ctk.CTkLabel(self, text="Press 'Start Training' to begin.", font=("Minecraftia", 16))
            self.status_label.grid(row=1, column=0, padx=20, pady=20)

            self.model = None


      def start_training(self):
            self.status_label.configure(text="Training in progress... The window will be unresponsive.")
            self.train_button.configure(state="disabled", text="Training...")
            self.back_button.configure(state="disabled")
            self.update_idletasks() 

            try: 
                  tomato_dataset = CNNdataloader.load_tomato_dataset()
                  train_data,test_data,train_loader,test_loader = CNNdataloader.create_train_and_test_dataset(64,tomato_dataset)

                  self.model =CNNmodel.init_model("default model")

                  CNNmodel.train_test_model(train_loader,test_loader,self.model,5, 0.00005)
                  CNNmodel.save_model(self.model,"default model")

                  self.status_label.configure(text=f"Training complete. Model saved as 'default model.pth'")
            except Exception as e:
                  self.status_label.configure(text=f"An error occurred: {e}")
                  print(f"An error occurred during training: {e}")

            finally:
                  self.train_button.configure(state="normal", text="Start Training")
                  self.back_button.configure(state="normal")