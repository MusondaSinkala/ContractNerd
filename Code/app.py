import os
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import time
from datetime import datetime
import traceback

# Set the working directory to the project root (Code directory)
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

from base.clause_comparison import clause_comparison

app = Flask(__name__, template_folder = 'ui/templates')

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

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
    """Enhanced clause analysis endpoint with improved parsing and features"""
    try:
        # Input validation
        jurisdiction = request.form.get('jurisdiction')
        contract_type = request.form.get('contractType')
        contract_file = request.files.get('contract')

        if not jurisdiction or not contract_type:
            return jsonify({"error": "Please select both jurisdiction and contract type"}), 400

        # File handling
        contract_path = None
        if contract_file and allowed_file(contract_file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            contract_filename = secure_filename(contract_file.filename)
            contract_path = os.path.join(app.config['UPLOAD_FOLDER'], contract_filename)
            contract_file.save(contract_path)
            print(f"Processing contract: {contract_path}")

        # Prepare legal references
        legal_resources = {
            'regulations': os.path.join(project_root, 'Data', 'Regulations',
                                        contract_type, jurisdiction, 'regulations.pdf'),
            'risky_clauses': os.path.join(project_root, 'Data', 'Risky Clauses',
                                          contract_type, jurisdiction, 'risky_clauses.txt')
        }

        # Enhanced analysis pipeline
        analysis_start = time.time()
        final_evaluation = clause_comparison(
            contract_path = contract_path,
            law_path      = legal_resources['regulations'],
            risky_clauses = legal_resources['risky_clauses'],
            model         = 'Meta-Llama-3.3-70B-Instruct',
            role          = "user",
            api_key       = "893bd5f1-b41e-4d17-ab1d-3ee3c7cba82b",
            api_base      = "https://api.sambanova.ai/v1",
            temperature   = 0.1,
            top_p         = 0.1,
            max_tokens    = 8192
        )
        print(f"Analysis completed in {time.time() - analysis_start:.2f}s")
        print(f"Final Evaluation: {final_evaluation}")

        # Parse enhanced output format
        clauses = []
        current_clause = None
        current_field = None

        for line in final_evaluation.split('\n'):
            line = line.strip()
            if not line:
                continue

            # New clause detection
            if re.match(r'^\d+\.', line):
                if current_clause:
                    clauses.append(current_clause)

                parts = line.split('.', 1)
                current_clause = {
                    'number': parts[0].strip(),
                    'text': parts[1].replace('Clause:', '').strip().strip('"'),
                    'classification': None,
                    'risk_tier': None,
                    'details': {
                        'regulations': None,
                        'linguistic_traits': None,
                        'explanation': None,
                        'improvement_guidance': None
                    }
                }
            elif current_clause:
                # Field detection with more robust parsing
                if line.startswith('Regulation(s) Implicated:'):
                    current_clause['details']['regulations'] = line.split(':', 1)[1].strip()
                elif line.startswith('Classification:'):
                    classification = line.split(':', 1)[1].strip()
                    current_clause['classification'] = classification
                    current_clause['is_unenforceable'] = 'Unenforceable' in classification
                elif line.startswith('Risk Tier:'):
                    current_clause['risk_tier'] = line.split(':', 1)[1].strip()
                elif line.startswith('Linguistic Traits Identified:'):
                    current_clause['details']['linguistic_traits'] = line.split(':', 1)[1].strip() or 'None listed'
                elif line.startswith('Explanation of Classification:'):
                    current_clause['details']['explanation'] = line.split(':', 1)[1].strip()
                elif line.startswith('Improvement Guidance:'):
                    current_clause['details']['improvement_guidance'] = line.split(':', 1)[1].strip()
                elif current_field:  # Handle multi-line fields
                    current_clause['details'][current_field] += '\n' + line

        # Add final clause if exists
        if current_clause:
            clauses.append(current_clause)

        # Generate analysis metadata
        analysis_metadata = {
            'jurisdiction': jurisdiction,
            'contract_type': contract_type,
            'timestamp': datetime.now().isoformat(),
            'clause_count': len(clauses),
            'unenforceable_count': sum(1 for c in clauses if c.get('is_unenforceable'))
        }

        return jsonify({
            "metadata": analysis_metadata,
            "clauses": clauses,
            "raw": final_evaluation,
            "legal_resources": legal_resources
        })

    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Analysis failed",
            "message": str(e),
            "trace": traceback.format_exc() if app.debug else None
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
