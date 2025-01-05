import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Hugging Face token
HF_TOKEN = "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"

# Load GPT-2
@st.cache_resource
def load_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Load Llama 3.2
@st.cache_resource
def load_llama():
    model = AutoModelForCausalLM.from_pretrained(
        "AmmarA22/Llama-3.2-1B-Instruct", use_auth_token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "AmmarA22/Llama-3.2-1B-Instruct", use_auth_token=HF_TOKEN
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text for GPT-2
def generate_gpt2(prompt, model, tokenizer, device, max_length=600, temperature=0.7, top_p=0.9):
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

# Generate text for Llama 3.2
def generate_llama(prompt, pipe, max_length=600):
    response = pipe(prompt, max_length=max_length, num_return_sequences=1)
    return response[0]["generated_text"]

# Streamlit UI
st.title("Multi-Model Text Generation")

# Dropdown for model selection
model_selection = st.selectbox("Select a Model", ["GPT-2", "Llama 3.2"])

# Input prompt
prompt = st.text_area("Enter your prompt", "What is AI?")

# Generate button
if st.button("Generate Text"):
    if model_selection == "GPT-2":
        model, tokenizer, device = load_gpt2()
        with st.spinner('Generating text with GPT-2...'):
            try:
                output = generate_gpt2(prompt, model, tokenizer, device)
                st.subheader("Generated Text (GPT-2):")
                st.write(output)
            except Exception as e:
                st.error(f"Error generating text with GPT-2: {e}")

    elif model_selection == "Llama 3.2":
        with st.spinner('Generating text with Llama 3.2...'):
            try:
                llama_pipe = load_llama()
                output = generate_llama(prompt, llama_pipe)
                st.subheader("Generated Text (Llama 3.2):")
                st.write(output)
            except Exception as e:
                st.error(f"Error generating text with Llama 3.2: {e}")
