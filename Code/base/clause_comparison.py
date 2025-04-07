import pickle
import openai
import json
from base.utils.functions import read_pdf_pymupdf, extract_info  # If publishing the code using flask
# from utils.functions import read_pdf_pymupdf, extract_info  # If testing locally

def clause_comparison(contract_path, law_path, risky_clauses, model, role, api_base, api_key, temperature, top_p, max_tokens, retries = 5):
    # Define Llama client
    client = openai.OpenAI(
        api_key  = api_key,
        base_url = api_base,
    )

    # Function to compare contract against relevant laws
    def law_comparison(contract, laws):
        prompt = f""" You are a contract language specialist reviewing contract clauses against given regulations.
                      Follow these steps:
                      1. Break down regulations into main requirements, sub-requirements, and compliance criteria.
                      2. Analyze each clause:
                         - Identify key components.
                         - Map components to regulatory requirements.
                         - Document potential violations.
                      3. Classify compliance or non-compliance:
                         - List the clause and relevant regulation.
                         - Explain why it may be non-compliant if appropriate.
                         - Assess confidence level (High/Medium/Low).
                      4. Self-check: Challenge assumptions, consider alternative interpretations, and document uncertainties.
                      5. Output the analysis of all clauses - including those which do not violate regulations - in the following format:
                         - Clause text
                         - Violated regulation
                         - Reasoning
                         - Confidence level                    
                      Contract Clauses:
                      {contract}
                
                      Regulations:
                      {laws[:1000]}
                   
                      Remember: If there is significant uncertainty about non-compliance, err on the side of caution and document the uncertainty rather than classifying as non-compliant.
                 """
        # 5. Output only confirmed violations, including:
        #    - Clause text
        #    - Violated regulation
        #    - Reasoning
        #    - Confidencelevel
        # Contract Clauses:
        # {contract[:1000]}

        response = client.chat.completions.create(
            model       = model,
            messages    = [{"role": role, "content": prompt}],
            temperature = temperature,
            top_p       = top_p,
            max_tokens  = max_tokens,
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content

    # Function to conduct few shot learning to classify "risky" clauses
    def few_shot_learning(comparison, risky_clauses_text):
        prompt = f"""You are a contract language specialist.
                     Having compared a contract against a set of regulations, you've identified the following:
                     {comparison}

                     Your goal is to classify each clause as strictly one of the following:
                     - Unenforceable
                     - Risky (but enforceable)
                     - Non-risky Enforceable

                     Review the output and compare it against the provided examples of risky clauses.

                     Follow this process step by step:

                     1. Risky Clause Comparison:
                        Compare all clauses classified with medium and low confidence against the provided examples of risky clauses. For each comparison:
                        - State whether the clause aligns with the examples of risky clauses.
                        - Provide justification for your comparison

                     2. Final Output:
                        The final output should classify clauses as unenforceable, risky, or non-risky enforceable after the analysis. For all clause, include:
                        - The full text of the clause.
                        - The exact regulation(s) or criteria implicated (if classified as unenforceable).
                        - It's classification (i.e., unenforceable, risky or non-risky enforceable)
                        - A clear explanation of its classification.
                        - Confidence level and key deciding factors.

                     Risky Clause examples:
                     {risky_clauses_text}

                     Give the final output for all clauses - Risky enforceable, Non-risky enforceable and unenforceable. Include all clauses highlighted in the comparison mentioned at the beginning of the prompt."""
                     # Remember: If there is significant uncertainty about classification, document the uncertainty rather than making a definitive classification."""

        response = client.chat.completions.create(
            model       = model,
            messages    = [{"role": role, "content": prompt}],
            temperature = temperature,
            top_p       = top_p,
            max_tokens  = max_tokens,
        )
        return response.choices[0].message.content

    # def few_shot_learning(comparison, risky_clauses_text):
    #     prompt = f"""You are a contract language specialist.
    #                  Having compared a contract against a set of regulations, you've identified the following:
    #                  {comparison}
    #
    #                  Your goal is to classify each clause strictly as one of the following:
    #                  - Unenforceable
    #                  - Risky (but enforceable)
    #                  - Non-risky Enforceable
    #
    #                  Review the output and compare it against the provided examples of risky clauses.
    #
    #                  Follow this process step by step:
    #
    #                  1. Risky Clause Comparison:
    #                     Compare all clauses classified with medium and low confidence against the provided examples of risky clauses. For each comparison:
    #                     - State whether the clause aligns with the examples of risky clauses.
    #                     - Provide justification for your comparison.
    #
    #                  2. Final Output:
    #                     The final output should classify clauses as **unenforceable, risky, or non-risky enforceable** after the analysis. For each flagged clause, include:
    #                     - The clause title.
    #                     - The full text of the clause.
    #                     - The exact regulation(s) or criteria implicated (if classified as unenforceable).
    #                     - Its classification (i.e., unenforceable, risky, or non-risky enforceable).
    #                     - A clear explanation of its classification.
    #                     - Confidence level and key deciding factors.
    #
    #                  **Return the final output in JSON format ONLY, with the following structure:**
    #                  {{
    #                      "Unenforceable": [
    #                          {{
    #                              "Clause Title": "...",
    #                              "Full Text": "...",
    #                              "Implicated Regulation": "...",
    #                              "Classification": "Unenforceable",
    #                              "Explanation": "...",
    #                              "Confidence Level": "...",
    #                              "Key Deciding Factors": "..."
    #                          }},
    #                          ...
    #                      ],
    #                      "Risky (but enforceable)": [
    #                          {{
    #                              "Clause Title": "...",
    #                              "Full Text": "...",
    #                              "Implicated Regulation": "...",
    #                              "Classification": "Risky (but enforceable)",
    #                              "Explanation": "...",
    #                              "Confidence Level": "...",
    #                              "Key Deciding Factors": "..."
    #                          }},
    #                          ...
    #                      ],
    #                      "Non-risky Enforceable": [
    #                          {{
    #                              "Clause Title": "...",
    #                              "Full Text": "...",
    #                              "Classification": "Non-risky Enforceable",
    #                              "Explanation": "...",
    #                              "Confidence Level": "...",
    #                              "Key Deciding Factors": "..."
    #                          }},
    #                          ...
    #                      ]
    #                  }}
    #
    #                  Risky Clause examples:
    #                  {risky_clauses_text}
    #
    #                  **Do not include any other text, just return valid JSON.**"""
    #
    #     response = client.chat.completions.create(
    #         model=model,
    #         messages=[{"role": role, "content": prompt}],
    #         temperature=temperature,
    #         top_p=top_p,
    #         max_tokens=max_tokens,
    #     )
    #
    #     # Parse the response as JSON
    #     try:
    #         structured_output = json.loads(response.choices[0].message.content)
    #         return structured_output
    #     except json.JSONDecodeError:
    #         print("Error: Response was not valid JSON.")
    #         return None

    # Read the contract file
    contract_text = read_pdf_pymupdf(contract_path)

    # Unpack the list into a string
    risky_clauses_text = ""
    # for item in loaded_data:
    #     risky_clauses_text += item + ('\n\n' if 'Combination:' in item else '\n')

    # Read the regulation file
    regulations_text = read_pdf_pymupdf(law_path)

    # Extract clauses from the contract
    clauses = extract_info(
        document    = contract_text,
        prompt      = "Extract and list each clause in this contract. Keep only legally significant terms, obligations, and figures (e.g., rent amount, penalties, dates). Remove redundant wording and boilerplate text. Summarize each clause in less than 3 sentences. Do not include introductory statements, explanations, or personal commentary - Present the output as a numbered list.",
        client      = client,
        model       = model,
        role        = role,
        temperature = temperature,
        top_p       = top_p,
        max_tokens  = max_tokens,
    )

    # Extract regulations from law document
    regulations = extract_info(
        document    = regulations_text,
        prompt      = "Extract only legal provisions from Real Property Law (§) and case law related to rental rights and landlord obligations. List statutes verbatim but exclude procedural explanations. Omit background, history, and general guidelines—keep only enforceable laws and legal precedents.",
        client      = client,
        model       = model,
        role        = role,
        temperature = temperature,
        top_p       = top_p,
        max_tokens  = max_tokens
    )

    comparison1 = law_comparison(clauses, regulations)
    comparison2 = few_shot_learning(comparison1, risky_clauses_text)

    return comparison2

if __name__ == "__main__":
    final_evaluation  = clause_comparison(
        contract_path = "D:/Downloads/Academics/Capstone Project/Data/Gold Standards/Rental/New York/Contract 1.pdf",
        law_path      = "D:/Downloads/Academics/Capstone Project/Data/Regulations/Residential tenants’ rights guide.pdf",
        risky_clauses = "D:/Downloads/Academics/Capstone Project/Data/Risky Clauses/Rental/New York/risky_clauses.pkl",
        model         = 'Meta-Llama-3.1-70B-Instruct',
        role          = "user",
        api_key       = "893bd5f1-b41e-4d17-ab1d-3ee3c7cba82b",
        api_base      = "https://api.sambanova.ai/v1",
        temperature   = 0.1,
        top_p         = 1.0,
        max_tokens    = 4096
    )
    print(final_evaluation)