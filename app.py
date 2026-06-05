from flask import Flask, render_template, request, jsonify
from summarizer import summarize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def do_summarize():
    data = request.get_json(force=True)
    text = data.get('text', '')
    max_length = data.get('max_length', 120)
    min_length = data.get('min_length', 30)

    try:
        max_length = int(max_length)
        min_length = int(min_length)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid length values'}), 400

    if not text.strip():
        return jsonify({'error': 'Empty text'}), 400
    if min_length < 1 or max_length < 1 or min_length > max_length:
        return jsonify({'error': 'min_length must be positive and <= max_length'}), 400

    summary = summarize(text, max_length=max_length, min_length=min_length)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
