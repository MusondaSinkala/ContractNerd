import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from base.clause_comparison import clause_comparison

app = Flask(__name__, template_folder = 'ui/templates')

# Folder where uploaded files are stored
UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle clause analysis requests."""
    try:
        jurisdiction  = request.form.get('jurisdiction')
        contract_type = request.form.get('contractType')
        contract_file = request.files.get('contractFile')

        # Validate inputs
        if not jurisdiction or not contract_type:
            return jsonify({"error": "Please select both a jurisdiction and a contract type."}), 400

        # Validate uploaded contract file
        contract_path         = None
        if contract_file and allowed_file(contract_file.filename):
            contract_filename = secure_filename(contract_file.filename)
            contract_path     = os.path.join(app.config['UPLOAD_FOLDER'], contract_filename)

            # Ensure the uploads directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)
            contract_file.save(contract_path)

        # Map jurisdiction and contract type to the correct legal document
        legal_doc_path = f"D:/Downloads/Academics/Capstone Project/Data/Regulations/{contract_type}/{jurisdiction}/regulations.pdf"

        # Call clause_comparison function
        final_evaluation  = clause_comparison(
            contract_path = contract_path,
            law_path      = legal_doc_path,
            risky_clauses = "D:/Downloads/Academics/Capstone Project/Data/Risky Clauses/Rental/New York/risky_clauses.pkl",
            model         = 'Meta-Llama-3.1-70B-Instruct',
            role          = "user",
            api_key       = "893bd5f1-b41e-4d17-ab1d-3ee3c7cba82b",
            api_base      = "https://api.sambanova.ai/v1",
            temperature   = 0.1,
            top_p         = 1.0,
            max_tokens    = 4096
        )

        print(f"Contract Path: {contract_path}")
        print(f"Legal Document Path: {legal_doc_path}")
        print(f"Final Evaluation: {final_evaluation}")

        final_evaluation = final_evaluation.replace("\n", "<br>")

        return jsonify({"result": final_evaluation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
