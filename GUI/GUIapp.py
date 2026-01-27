import customtkinter as ctk
from PIL import Image
from . import GUItest, GUItrain, GUIeval



class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Classifier: Electrosphere")
        self.geometry("900x600")
        self.after(100, self.maximize)
    
        self.frames = {
            "menu": menu_frame(self),
            "train": GUItrain.training_frame(self),       # To be implemented
            "predict": GUItest.predict_image_frame(self),
            "eval" : GUIeval.eval_frame(self) #To be implemented

        }
        
        self.current_frame = self.frames["menu"]
        self.current_frame.pack(fill="both", expand=True)

    def maximize(self):
        self.state("zoomed")
     

    def show_frame(self, frame_name):
        self.current_frame.pack_forget()
        self.current_frame = self.frames[frame_name]
        self.current_frame.pack(fill="both", expand=True)




class menu_frame(ctk.CTkFrame):
    def __init__(self,master):
        super().__init__(master)

        for i in range(10):
            self.grid_rowconfigure(i, weight=1)
            self.grid_columnconfigure(i, weight=1)
    
        self.title_logo = ctk.CTkLabel(self, 
                                       text = "Plant Health Classifier",
                                         font= ctk.CTkFont("Minecraftia", size= 60))
        
        self.title_logo.grid(row=0, 
                             column=0, 
                             columnspan=10, 
                             rowspan=2,
                             padx=20, 
                             pady=20, 
                             sticky="nsew")

        self.train_btn = ctk.CTkButton(self, 
                                      text="Train Model", 
                                      font= ctk.CTkFont("Minecraftia", size= 30), 
                                      fg_color="transparent", 
                                      hover_color="#3A3A3A",
                                       command=lambda: master.show_frame("train") )
        self.train_btn.grid(row = 4,
                            column = 0,
                            columnspan = 10 )

        self.test_btn = ctk.CTkButton(self, 
                                    text="Test Model", 
                                    font= ctk.CTkFont("Minecraftia", size= 30), 
                                    fg_color="transparent", 
                                    hover_color="#3A3A3A",
                                    command=lambda: master.show_frame("predict") )
        self.test_btn.grid(row = 6,
                            column = 0,
                            columnspan = 10)

        # Added Statistic button
        self.statistic_btn = ctk.CTkButton(self, 
                                           text="Statistic", 
                                           font= ctk.CTkFont("Minecraftia", size= 30), 
                                           fg_color="transparent", 
                                           hover_color="#3A3A3A",
                                           command=lambda: master.show_frame("eval") )
        self.statistic_btn.grid(row = 8,
                                 column = 0,
                                 columnspan = 10)



def Init_App():
    ctk.set_appearance_mode("dark")  
    app = App()
    app.mainloop()