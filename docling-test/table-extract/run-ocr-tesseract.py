import logging
import time
from pathlib import Path
import json

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    # input_doc_path = Path("../pdfs/SIDARTHE paper.pdf")
    input_doc_path = Path("../pdfs/SIR paper 1.pdf")
    # input_doc_path = Path("../pdfs/SIR paper 2.pdf")
    output_dir = Path("output-ocr-tesseract")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_table_images = True

    # Any of the OCR options can be used:EasyOcrOptions, TesseractOcrOptions, TesseractCliOcrOptions, OcrMacOptions(Mac only), RapidOcrOptions
    # ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    # ocr_options = TesseractOcrOptions(force_full_page_ocr=True)
    # ocr_options = OcrMacOptions(force_full_page_ocr=True)
    # ocr_options = RapidOcrOptions(force_full_page_ocr=True)
    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options

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
        page_size = conv_res.document.pages[table.prov[0].page_no].size
        table_extract = {
            "page_no": table.prov[0].page_no,
            "page_dimensions": { "width": page_size.width, "height": page_size.height },
            "bbox": table.prov[0].bbox.as_tuple(),
            "coord_origin": table.prov[0].bbox.coord_origin,
            "text": table.export_to_html()
        }
        table_extracts.append(table_extract)
        table_html += table_extract["text"]

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