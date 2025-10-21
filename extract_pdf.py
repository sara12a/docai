import os
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        print(f"📄 Total pages: {num_pages}")

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += f"\n\n--- Page {i+1} ---\n\n"
            text += page_text if page_text else "[No text found]"
    return text

if __name__ == "__main__":
    # 🔹 Specify subfolder and file name
    subfolder = "pdfs"
    filename = "sample.pdf"
    pdf_path = os.path.join(subfolder, filename)

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
    else:
        output_text = extract_text_from_pdf(pdf_path)

        # Save extracted text to .txt file
        output_file = os.path.join(subfolder, filename.replace(".pdf", "_extracted.txt"))
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(output_text)
        print(f"✅ Text extracted and saved to {output_file}")
