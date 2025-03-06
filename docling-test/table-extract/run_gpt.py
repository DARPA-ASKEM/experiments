import logging
import time
from pathlib import Path
import json
import os
import io
import base64
from openai import OpenAI
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from PIL import Image

_log = logging.getLogger(__name__)

TABLE_EXTRACTION_PROMPT = """Please extract a table from the images and provide the table data formatted as an XHTML Table.

Some images may not contain tables and may only contain a mix of text, figures, graphs and equations. Please ignore these images and give them a score of 0.
Some images may contain a single table, while others may contain multiple tables. Please extract all tables present in the image.

Table may not be well-defined or are not easily extractable. Please do your best to use html tags and column and row spans to format the extracted table to align with the structure of the table in the image. Use visual cues to separate the columns and rows and to determine cells that span multiple columns. Ensure that symbols, subscripts, superscripts, and greek characters are preserved, do not swap "α" to "a" for example.

You will structure your response as a JSON object with the following schema:

'table_text': a XHTML formatted table string.
'score': A score from 0 to 10 indicating the quality of the extracted table. 0 indicates that the image does not contain a table, 10 indicates a high-quality extraction.

Begin:
"""

def image_to_base64_string(img: Image.Image) -> str:
    format = 'PNG'
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_base64}"  # Return Data URI

def process_table_image(image_uri):
    _log.info(f"Processing table image through gpt...")
    openai_api_key = os.getenv("OPEN_AI_API_KEY")
    if (openai_api_key is None):
        raise ValueError("OPEN_AI_API_KEY not found in environment variables. Please set 'OPEN_AI_API_KEY'.")

    client = OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TABLE_EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_uri},
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    print(response.choices[0].message.content)
    message_content = json.loads(response.choices[0].message.content)
    return message_content

def main():
    logging.basicConfig(level=logging.INFO)

    # input_doc_path = Path("../pdfs/SIR paper 1.pdf")
    # output_dir = Path("output-gpt")

    input_doc_path = Path("../pdfs/Measles.pdf")
    output_dir = Path("output-gpt-measles")

    pipeline_options = PdfPipelineOptions()
    # Needed to extract the table images
    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc_filename = conv_res.input.file.stem

    table_extracts = []
    table_html = ''
    # Export tables
    for table_ix, table in enumerate(conv_res.document.tables):
        # Get the table image
        table_img = table.get_image(conv_res.document).resize((1024, 1024)) # image with larger size seem to have better extraction results from GPT
        page_size = conv_res.document.pages[table.prov[0].page_no].size
        table_image_extract = process_table_image(image_to_base64_string(table_img))
        table_extract = {
            "page_no": table.prov[0].page_no,
            "page_dimensions": { "width": page_size.width, "height": page_size.height },
            "bbox": table.prov[0].bbox.as_tuple(),
            "coord_origin": table.prov[0].bbox.coord_origin,
            "text": table_image_extract["table_text"],
            "score": table_image_extract["score"]
        }
        table_extracts.append(table_extract)
        table_html += table_extract["text"]

        # Save the table image
        table_img_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.png"
        table_img.save(table_img_filename)
        _log.info(f"Saved table image to {table_img_filename}")

    output_filename = output_dir / f"{doc_filename}-tables.json"
    _log.info(f"Saving table extracts to {output_filename}")
    with output_filename.open("w", encoding="utf-8") as fp:
        json.dump(table_extracts, fp, indent=2, ensure_ascii=False)
    
    html_filename = output_dir / f"{doc_filename}-tables.html"
    _log.info(f"Saving table extracts to {html_filename}")
    with html_filename.open("w", encoding="utf-8") as fp:
        fp.write(table_html)

    end_time = time.time() - start_time

    _log.info(f"Document converted and tables exported in {end_time:.2f} seconds.")
  
if __name__ == "__main__":
    main()
