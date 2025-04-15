import fitz

def read_pdf_pymupdf(file_path):
    """
    Reads the contents of a PDF file and returns the extracted text.
    """
    text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text

def extract_info(document, prompt, client, model, role, temperature, top_p, max_tokens):
    """
    Extracts information from the given document using the specified LLM model.
    """
    response = client.chat.completions.create(
        model       = model,
        messages    = [{"role": role,
                        "content": f"{prompt}: {document}"}],
        temperature = temperature,
        top_p       = top_p,
        max_tokens  = max_tokens,
    )

    # Collect streamed output into a single variable
    text = response.choices[0].message.content

    return (text)