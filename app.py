import gradio as gr
from rag_pipeline import rag_answer

def chat_interface(user_input):
    answer, sources = rag_answer(user_input)
    
    # Format sources for display
    sources_text = "\n\n".join([f"Complaint ID: {s['complaint_id']} | Product: {s['product']}\n{s['snippet']}" for s in sources])
    
    return answer, sources_text

# Gradio Interface
iface = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about complaints..."),
    outputs=[
        gr.Textbox(label="RAG Answer"),
        gr.Textbox(label="Retrieved Sources")
    ],
    title="CrediTrust Complaint RAG Chatbot",
    description="Ask questions and get answers based on real customer complaints."
)

if name == "main":
    iface.launch()