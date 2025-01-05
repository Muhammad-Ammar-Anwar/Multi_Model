# Import necessary libraries
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Define HF token (if required for private models)
HF_TOKEN = "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"

# Load models and tokenizers
@st.cache_resource
def load_gpt_neo():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

@st.cache_resource
def load_llama():
    return pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer="meta-llama/Llama-3.2-1B-Instruct",  # Match tokenizer and model
        device=0 if torch.cuda.is_available() else -1,
    )

# Define text generation functions
def generate_text_gpt_neo(prompt, model, tokenizer, device, max_length=100, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_text_llama(prompt, pipe, max_length=600):
    response = pipe(prompt, max_length=max_length, num_return_sequences=1)
    return response[0]["generated_text"]

# Streamlit UI
st.title("Text Generation with Llama 3.2 and GPT Neo")

# Model Selection
model_selection = st.selectbox("Select a model", ["GPT Neo", "Llama 3.2"])

# Input Prompt
prompt = st.text_area("Enter your prompt", "What is chest pain?")

# Generate Text
if st.button("Generate Text"):
    if model_selection == "GPT Neo":
        st.write("Loading GPT Neo model...")
        model, tokenizer, device = load_gpt_neo()
        st.write("Generating text with GPT Neo...")
        output = generate_text_gpt_neo(prompt, model, tokenizer, device)
        st.subheader("Generated Text:")
        st.write(output)
    elif model_selection == "Llama 3.2":
        st.write("Loading Llama 3.2 model...")
        llama_pipe = load_llama()
        st.write("Generating text with Llama 3.2...")
        output = generate_text_llama(prompt, llama_pipe)
        st.subheader("Generated Text:")
        st.write(output)
