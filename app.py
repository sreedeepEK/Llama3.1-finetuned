import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sreedeepEK/lora_model")
model = AutoModelForCausalLM.from_pretrained("sreedeepEK/lora_model")

def generate_text(input_text):
    # Prepare the input
    inputs = tokenizer.encode_plus(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask") 
    # Generate text
    outputs = model.generate(input_ids, max_length=50, attention_mask=attention_mask)
    
    # Decode and return the output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit interface
st.title("LlaMA_3.1 Finetuned!")
st.write("Enter a prompt to generate text based on the provided input.")

input_text = st.text_area("Enter your text:", "")

if st.button("Generate output"):
    if input_text:
        output_text = generate_text(input_text)
        st.text_area("Generated text:", value=output_text, height=200)
    else:
        st.write("Please enter some text to generate output.")
