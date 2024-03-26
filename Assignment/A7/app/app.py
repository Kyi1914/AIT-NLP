# app.py

from flask import Flask, render_template, request, jsonify
from gpt import answer_question

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # user_message = request.form['user_message']
    # return jsonify({'bot_response': user_message})
    if request.method == "POST":
        # get user message
        user_message = request.form.get('user_message', '')  
        # Process the user's message and generate a response
        generated_response = answer_question(user_message)  
        return render_template('index.html', generated_response=generated_response)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
