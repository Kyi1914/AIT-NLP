from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the model and tokenizer
model_name_or_path = "distilgpt2"
# model_name_or_path = 'load my save model and load the model'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Define the home page route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input from the form
        prompt = request.form['prompt']
        
        # Tokenize the input text
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate output text
        output = model.generate(input_ids, max_length=256, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
        
        # Decode and format the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return render_template('index.html', prompt=prompt, generated_text=generated_text)
    
    # Render the home page template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
