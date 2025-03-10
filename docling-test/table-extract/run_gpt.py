import logging
import time
from pathlib import Path
import json
import os
import io
import base64
from openai import OpenAI
from docling.datamodel.base_models import InputFormat
from bs4 import BeautifulSoup
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from PIL import Image

_log = logging.getLogger(__name__)

TABLE_EXTRACTION_PROMPT = """Here is the extracted table in html from the provided image.

The extracted result isn't perfect, but it's a good start. You know that the structure of the table is correct, but the extracted content of the each cells may not be accurate or be missing.

Your job is to verify the extracted table content with the provided image and correct any inaccuracies or missing data. Do not alter the structure, colspan, or rowspan of the table, only replace the cell text value with the correct data. Some cells may span multiple columns or rows which is normal and should be preserved. Also it's normal to have multiple header rows or columns. Do not remove any rows or columns from the table, just correct the cell text values. Number of rows and columns should be preserved. 
Ensure that symbols, subscripts, superscripts, and greek characters are preserved, "α" should not be swapped to "a". Make sure symbols, greek characters, and mathematical expressions are preserved and represented correctly.

You will structure your response as a JSON object with the following schema:

'table_text': a XHTML formatted table string.
'score': A score from 0 to 10 indicating the quality of the corrected table. 0 indicates that the image does not contain a table, 10 indicates a high-quality extraction.

Begin:
"""

IMAGE_RESOLUTION_SCALE = 2.0

def extract_table_grid(html_table):
    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')
    
    # Determine the size of the table
    num_rows = len(table.find_all('tr'))
    num_cols = max(len(row.find_all(['td', 'th'])) for row in table.find_all('tr'))
    
    # Initialize the grid with empty strings
    grid = [['' for _ in range(num_cols)] for _ in range(num_rows)]
    
    for row_idx, row in enumerate(table.find_all('tr')):
        col_idx = 0
        for cell in row.find_all(['td', 'th']):
            while grid[row_idx][col_idx]:
                col_idx += 1
            
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            cell_text = cell.get_text(strip=True)
            
            for i in range(rowspan):
                for j in range(colspan):
                    grid[row_idx + i][col_idx + j] = cell_text
            
            col_idx += colspan
    
    return grid

def image_to_base64_string(img: Image.Image) -> str:
    format = 'PNG'
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_base64}"  # Return Data URI

def process_table_image(image_uri, table_html):
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
                    {
                        "type": "image_url",
                        "image_url": {"url": image_uri},
                    },
                    {
                        "type": "text",
                        "text": table_html,
                    },
                    {"type": "text", "text": TABLE_EXTRACTION_PROMPT},
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

    # input_doc_path = Path("../pdfs/Rosenblatt_2024.pdf")
    # output_dir = Path("output-gpt-rosenblatt")

    # input_doc_path = Path("../pdfs/SIR paper 1.pdf")
    # output_dir = Path("output-gpt")

    input_doc_path = Path("../pdfs/Measles.pdf")
    output_dir = Path("output-gpt-measles")

    pipeline_options = PdfPipelineOptions()
    # Needed to extract the table images
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = RapidOcrOptions(force_full_page_ocr=True)

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
        table_img = table.get_image(conv_res.document)
        table_img = table_img.resize((table_img.width * 2, table_img.height * 2))
        page_size = conv_res.document.pages[table.prov[0].page_no].size
        table_image_extract = process_table_image(image_to_base64_string(table_img), table.export_to_html())
        html_table = table_image_extract["table_text"]

        num_rows = table.data.num_rows
        num_cols = table.data.num_cols
        table_grid = extract_table_grid(html_table)
        table_cells = []
        # If original table and gpt extracted table has same dimension, update each cell value with the extracted table data
        if (num_rows == len(table_grid) and num_cols == len(table_grid[0])):
            # update table cell values
            for table_cell in table.data.table_cells:
                row_idx = table_cell.start_row_offset_idx
                col_idx = table_cell.start_col_offset_idx
                cell_val = table_grid[row_idx][col_idx]
                table_cell.text = str(cell_val)
                cell_dict = vars(table_cell)
                cell_dict["bbox"] = table_cell.bbox.as_tuple()
                table_cells.append(cell_dict)

        table_extract = {
            "page_no": table.prov[0].page_no,
            "page_dimensions": { "width": page_size.width, "height": page_size.height },
            "bbox": table.prov[0].bbox.as_tuple(),
            "coord_origin": table.prov[0].bbox.coord_origin,
            "text": html_table,
            "table_cells": table_cells
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
