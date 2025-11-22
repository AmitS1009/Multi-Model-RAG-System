import fitz
from PIL import Image
import pytesseract
import io
import os


class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)


    #  TEXT EXTRACTION

    def extract_text_chunks(self):
        chunks = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()

            if text.strip():
                chunks.append({
                    'type': 'text',
                    'content': text.strip(),
                    'page': page_num + 1,
                    'source': f'Page {page_num + 1}'
                })

        return chunks


    #  TABLE EXTRACTION

    def extract_tables(self):
        tables = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            try:
                blocks = page.get_text("dict")["blocks"]
            except:
                continue

            for block in blocks:
                # If block has multiple lines - treating them as potential table
                if "lines" in block and len(block["lines"]) >= 3:
                    table_text = ""

                    for line in block["lines"]:
                        for span in line["spans"]:
                            table_text += span["text"] + " "
                        table_text += "\n"

                    if table_text.strip():
                        tables.append({
                            'type': 'table',
                            'content': table_text.strip(),
                            'page': page_num + 1,
                            'source': f'Table on Page {page_num + 1}'
                        })

        return tables


    #  IMAGE EXTRACTION + OCR

    def extract_images_with_ocr(self, output_folder=None):
        
        if output_folder is None:
            try:
                import config
                output_folder = config.IMAGES_DIR
            except:
                output_folder = "extracted_images"

        os.makedirs(output_folder, exist_ok=True)

        images_data = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]

                try:
                    base_image = self.doc.extract_image(xref)
                except:
                    continue

                image_bytes = base_image.get("image")

                # Save extracted image
                image_filename = f"{output_folder}/page{page_num+1}_img{img_index+1}.png"
                with open(image_filename, "wb") as f:
                    f.write(image_bytes)

                # OCR
                try:
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(pil_img)
                except Exception as e:
                    print(f"OCR failed on Page {page_num + 1}: {e}")
                    continue

                if ocr_text.strip():
                    images_data.append({
                        'type': 'image',
                        'content': ocr_text.strip(),
                        'page': page_num + 1,
                        'image_path': image_filename,
                        'source': f'Image on Page {page_num + 1}'
                    })

        return images_data


    #  FULL PIPELINE

    def process_document(self):
        print(f"Processing document: {self.pdf_path}")

        text_chunks = self.extract_text_chunks()
        print(f"Extracted {len(text_chunks)} text chunks")

        tables = self.extract_tables()
        print(f"Extracted {len(tables)} tables")

        images = self.extract_images_with_ocr()
        print(f"Extracted {len(images)} images with OCR")

        all_chunks = text_chunks + tables + images
        print(f"Total chunks: {len(all_chunks)}")

        return all_chunks

    def close(self):
        self.doc.close()



#   RUN DIRECTLY

if __name__ == "__main__":
    
    import config

    processor = DocumentProcessor(config.PDF_PATH)
    chunks = processor.process_document()

    if chunks:
        print("\nSample chunk:\n", chunks[0])

    processor.close()
