import customtkinter as ctk
from PIL import Image

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("uhhhhhhhhhh")
        self.geometry("800x600")      
        
        self.frame = menu_frame(master =self)
        self.frame.pack(fill="both", expand=True)



class menu_frame(ctk.CTkFrame):
    def __init__(self,master):
        super().__init__(master)

        for i in range(10):
            self.grid_rowconfigure(i, weight=1)
            self.grid_columnconfigure(i, weight=1)
    
        self.title_logo = ctk.CTkLabel(self, 
                                       text = "Blue Archive",
                                         font= ctk.CTkFont("Minecraftia", size= 60))
        
        self.title_logo.grid(row=0, 
                             column=0, 
                             columnspan=10, 
                             rowspan=2,
                             padx=20, 
                             pady=20, 
                             sticky="nsew")

        self.btn_opt1 = ctk.CTkButton(self, 
                                      text="Train Model", 
                                      font= ctk.CTkFont("Minecraftia", size= 30), 
                                      fg_color="transparent", 
                                      hover_color="#3A3A3A" )
        self.btn_opt1.grid(row = 4,
                            column = 0,
                            columnspan = 10 )

        self.btn_opt2 = ctk.CTkButton(self, 
                                      text="Test Model", 
                                      font= ctk.CTkFont("Minecraftia", size= 30), 
                                      fg_color="transparent", 
                                      hover_color="#3A3A3A" )
        self.btn_opt2.grid(row = 6,
                            column = 0,
                            columnspan = 10)


class predict_image_frame(ctk.CTkFrame):
    def __init__(self,master):
        super().__init__(master)

        for i in range(10):
            self.grid_rowconfigure(i, weight=1)
            self.grid_columnconfigure(i, weight=1)
    



app =App()
app.mainloop()