from src.data_processing.vector_store import initialize_vector_store
from src.gradio_interface.chat_ui import create_chat_interface
import gradio as gr
import signal
import sys

def graceful_exit(signum, frame):
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, graceful_exit)  
    
    vector_store = initialize_vector_store()
    demo = create_chat_interface(vector_store)
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            prevent_thread_lock=True,  
            show_error=True
        )
        while True:
            pass 
    except KeyboardInterrupt:
        
        print("")
    finally:
        demo.close()

if __name__ == "__main__":
    main()