import tkinter as tk
from tkinter import ttk, scrolledtext
from pdf_qa import create_faiss_index, search_chunks, ask_ai_about_pdf, ALL_MODELS
import numpy as np
import threading

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF QA Chatbot")
        self.root.geometry("800x600")
        
        # Create loading indicator
        self.loading_label = None
        
        # Initialize chat history
        self.chat_history = []
        
        # Initialize the QA system
        self.embeddings = np.load('embeddings.npy')
        self.index = create_faiss_index(self.embeddings)
        
        # Load chunks from the PDF
        with open('chunks.txt', 'r', encoding='utf-8') as f:
            self.chunks = f.read().split('\n===\n')
        
        # Initialize selected model
        self.selected_model = tk.StringVar()
        self.selected_model.set('gpt-4o_v2024-05-13_NOFILTER_GaaS')  # Set default model
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Create model selection dropdown
        model_frame = ttk.Frame(main_container)
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            values=ALL_MODELS,
            state="readonly",
            width=50
        )
        model_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create and configure chat display
        self.chat_display = scrolledtext.ScrolledText(
            main_container,
            wrap=tk.WORD,
            width=70,
            height=30,
            font=("Arial", 10)
        )
        self.chat_display.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chat_display.config(state=tk.DISABLED)
        
        # Configure text tags for alignment and colors
        self.chat_display.tag_configure("right", justify="right", foreground="#00008B")  # Dark Blue
        self.chat_display.tag_configure("left", justify="left")
        
        # Create loading indicator label
        self.loading_label = ttk.Label(main_container, text="Thinking...", font=("Arial", 10))
        self.loading_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        self.loading_label.grid_remove()  # Hide initially
        
        # Create input field
        self.input_field = ttk.Entry(
            main_container,
            width=60,
            font=("Arial", 10)
        )
        self.input_field.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=(0, 5), pady=10)
        self.input_field.bind("<Return>", self.send_message)
        
        # Create send button
        self.send_button = ttk.Button(
            main_container,
            text="Send",
            command=self.send_message
        )
        self.send_button.grid(row=2, column=1, sticky=(tk.E), pady=10)
        
        # Configure grid weights
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=0)
        main_container.rowconfigure(0, weight=1)
        
        # Add initial message
        self.display_message("System: Welcome! Ask me questions about the PDF document.", "system")
        
    def display_message(self, message, sender):
        self.chat_display.config(state=tk.NORMAL)
        if sender == "user":
            self.chat_display.insert(tk.END, f"\n{message}\n", "right")
        else:
            self.chat_display.insert(tk.END, f"\n{message}\n", "left")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self, event=None):
        question = self.input_field.get().strip()
        if not question:
            return
        
        # Clear input field
        self.input_field.delete(0, tk.END)
        
        # Display user message
        self.display_message(f"You: {question}\n", "user")
        
        # Disable input and show loading indicator
        self.input_field.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.loading_label.grid()  # Show loading indicator
        self.chat_display.see(tk.END)  # Ensure loading indicator is visible
        
        # Process in separate thread
        threading.Thread(target=self.process_question, args=(question,), daemon=True).start()
    
    def process_question(self, question):
        try:
            # Search for relevant chunks
            relevant_chunks = search_chunks([question], self.chunks, self.index, self.embeddings)
            
            # Get AI response
            answer = ask_ai_about_pdf(relevant_chunks, question, self.chat_history, model=self.selected_model.get())
            
            # Update chat history
            self.chat_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
            
            # Keep only last 20 messages
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            
            # Display response
            self.root.after(0, self.display_message, f"Assistant: {answer}", "assistant")
        except Exception as e:
            self.root.after(0, self.display_message, f"Error: {str(e)}", "system")
        finally:
            # Re-enable input and hide loading indicator
            self.root.after(0, lambda: self.input_field.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
            self.root.after(0, self.loading_label.grid_remove)  # Hide loading indicator
            self.root.after(0, self.input_field.focus)

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
