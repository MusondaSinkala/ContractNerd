# ContractNerd
An AI-powered tool for analyzing contracts and identifying risky clauses.

### Project Overview

Contract Nerd is a contract analysis tool designed to detect and highlight potential risks in legal agreements. The tool identifies missing elements, illegal terms, and unfair clauses that could be detrimental to the second party in a contract. It consists of both a front-end UI for user interaction and a back-end system for processing contract data.

### Features
- Clause Risk Analysis: Detects ambiguous, overly general, redundant, and unconscionable clauses.
- Machine Learning Models: Uses NLP techniques to evaluate contract fairness.
- Front-end Interface: A simple HTML-based UI for users to upload and analyze contracts.
- API Integration: The back-end processes requests and serves results via API calls.

### Folder Structure

The repository is organized as follows:
```
ContractNerd
│── Code/                        # Source code for the project
│   ├── base/                    # Core contract analysis logic
│   │   ├── utils/               # Utility code
│   │   │   ├── functions.py     
│   │   ├── clause_comparison.py # Code to compare contract clauses against regulations 
│   │   ├── clause_generation.py # Legacy code to generate risky clauses using few-shot learning
│   │   ├── main.py              
│   ├── ui/                      # Front-end interface
│   │   ├── templates/           # HTML files
│   │   │   │── archive/         # Old or unused UI components (ignored in Git)
│   │   │   │── about.html       # About page
│   │   │   │── index.html       # Home page
|   ├── static/                  # Image assets
|   ├── app.py                   # Flask implementation of app
│── Data/                        # Data used for contract analysis
│   ├── Risky Clauses/           # Example contract clauses flagged as risky (ignored in Git) - used for few-shot learning
│   ├── Contracts/               
│   ├── Gold Standards/          # Sample contracts with enforcable clauses
│   ├── Regulations/             # Regulation documentation
│   ├── Tests/           
│── venv/                        # Virtual environment (ignored in Git)
│── requirements.txt             # List of Python dependencies
│── .gitignore                   # Files and directories ignored by Git
│── README.md                    # Project documentation
```

### Installation Instructions

1. Clone the repository:
   - git clone https://github.com/MusondaSinkala/ContractNerd.git
   - cd ContractNerd
2. Create and Activate a Virtual Environment
   - python -m venv venv
   - On Windows: venv\Scripts\activate
   - On macOS/Linux: source venv/bin/activate
3. Install dependencies
   - pip install -r requirements.txt

### Usage
1. Start the back-end API:
   - python Code/base/app.py
2.  Open the front-end UI in a browser
