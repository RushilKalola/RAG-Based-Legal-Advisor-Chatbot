from pypdf import PdfReader


class PDFLoader:
    def load(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text