import transformers
import streamlit as st
import time

@st.cache(allow_output_mutation=True)
def load_models():
    # English to Tagalog
    EN_TL_MODEL = "Helsinki-NLP/opus-mt-en-tl"
    en_tl_tokenizer = transformers.AutoTokenizer.from_pretrained(EN_TL_MODEL)
    en_tl_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(EN_TL_MODEL)
    en_tl_translator = transformers.pipeline("text2text-generation", model=en_tl_model, tokenizer=en_tl_tokenizer, device=0)
    # Tagalog to English
    TL_EN_MODEL = "Helsinki-NLP/opus-mt-tl-en"
    tl_en_tokenizer = transformers.AutoTokenizer.from_pretrained(TL_EN_MODEL)
    tl_en_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(TL_EN_MODEL)
    tl_en_translator = transformers.pipeline("text2text-generation", model=tl_en_model, tokenizer=tl_en_tokenizer, device=0)
    return en_tl_translator, tl_en_translator

en_tl_translator, tl_en_translator = load_models()

def en_to_tl(input_text):
    input_text = input_text.strip()
    if len(input_text) > 0:
        transation = en_tl_translator(input_text)[0]["generated_text"]
        return transation.strip()
    else:
        return ""

def tl_to_en(input_text):
    input_text = input_text.strip()
    if len(input_text) > 0:
        transation = tl_en_translator(input_text)[0]["generated_text"]
        return transation.strip()
    else:
        return ""

st.title("English-Tagalog Translation App")

direction = st.selectbox("Direction", ["English -> Tagalog", "Tagalog -> English"])

input_text = st.text_area("Input text", value="")

start_time = time.time()
if direction == "English -> Tagalog":
    translation = en_to_tl(input_text)
else:
    translation = tl_to_en(input_text)
end_time = time.time()

time_taken = str(round(end_time-start_time,2))

st.markdown("**Translation**: "+translation)

st.markdown("Time taken: "+str(time_taken)+"s")

