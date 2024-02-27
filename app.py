from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load the saved model and tokenizer
model_path = "model"
tokenizer_path = "tokenizer"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Translation function
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    translation = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
    return translated_text

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        input_text = request.form['input_text']
        translated_text = translate_text(input_text)
        return render_template('index.html', input_text=input_text, translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
