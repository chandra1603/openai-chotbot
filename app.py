import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.chat.util import Chat, reflections
import random
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Download NLTK data
nltk.download('punkt')

# Load Transformer Model
print("Loading AI Model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
chat_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Model Loaded Successfully!")

# GUI Chatbot Application
class ChatBotApp:
    def __init__(self, master):
        self.master = master
        master.title("AI ChatBot")
        master.geometry("600x500")
        master.configure(bg="#000000")  # Black background
        
        # Chat history display
        self.chat_history = scrolledtext.ScrolledText(
            master, width=60, height=20, bg="#000000", fg="#FFFFFF", font=("Arial", 12)
        )
        self.chat_history.pack(padx=10, pady=10)
        self.chat_history.insert(tk.END, "Bot: Hello! How can I help you today?\n")
        self.chat_history.config(state=tk.DISABLED)
        
        # Input frame
        self.input_frame = tk.Frame(master, bg="#000000")
        self.input_frame.pack(padx=10, pady=10, fill=tk.X)
        
        # User input box
        self.user_input = tk.Entry(self.input_frame, width=50, bg="#222222", fg="#FFFFFF", font=("Arial", 12))
        self.user_input.pack(side=tk.LEFT, padx=5)
        self.user_input.bind("<Return>", self.send_message)
        
        # Send button
        self.send_button = tk.Button(
            self.input_frame, text="Send", command=self.send_message, 
            bg="#333333", fg="#FFFFFF", font=("Arial", 12)
        )
        self.send_button.pack(side=tk.LEFT, padx=5)
        
    def send_message(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        
        self._display_message(f"You: {user_text}\n")
        self.user_input.delete(0, tk.END)
        
        try:
            chat_response = self._generate_response(user_text)
        except Exception as e:
            chat_response = "I'm having trouble thinking right now. Please try again later."
        
        self._display_message(f"Bot: {chat_response}\n")
    
    def _generate_response(self, text):
        inputs = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return response
    
    def _display_message(self, message):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message)
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatBotApp(root)
    root.mainloop()

