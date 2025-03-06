import json
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)



from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


################################################################################
# Very basic vanilla test for PDF extractions
################################################################################
# source = "/Users/dchang/workspace/askem/pdfs/SIDARTHE_Giordano2020.pdf"
source = "pdfs/SIDARTHE paper.pdf"
source = "pdfs/Measles.pdf"

logger = logging.getLogger(__name__)


# Options
# Important: For operating with page images, we must keep them, otherwise the DocumentConverter
# will destroy them for cleaning up memory.
# This is done by setting PdfPipelineOptions.images_scale, which also defines the scale of images.
# scale=1 correspond of a standard 72 DPI image
# The PdfPipelineOptions.generate_* are the selectors for the document elements which will be enriched
# with the image field
IMAGE_RESOLUTION_SCALE = 2.0
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
result = converter.convert(source)

# Save images of figures and tables
output_dir = Path("scratch")
output_dir.mkdir(parents=True, exist_ok=True)

doc_filename = result.input.file.stem

picture_counter = 0


# Dump text formulas
formula_counter = 0
for _idx, element in enumerate(result.document.texts):
    # if isinstance(element, TextItem):
    #     logger.info(f"{element.label} =>  {element.text}")

    if element.label == "formula":
        logger.info(f"{element.label} =>  {element.text}")
        text_img = element.get_image(result.document)
        logger.info(text_img)

        formula_counter += 1
        element_image_filename = (
            output_dir / f"{doc_filename}-formula-{formula_counter:03}.png"
        )
        with element_image_filename.open("wb") as fp:
            text_img.save(fp, "PNG")


# Dump tables
table_counter = 0
for _idx, element in enumerate(result.document.tables):
    logger.info("table...")
    table_img = element.get_image(result.document)
    table_counter += 1
    element_image_filename = (
        output_dir / f"{doc_filename}-table-{table_counter}.png"
    )
    with element_image_filename.open("wb") as fp:
        table_img.save(fp, "PNG")

"""
for element, _level in result.document.iterate_items():
    logger.info("processing document item")
    if isinstance(element, TableItem):
        table_counter += 1
        element_image_filename = (
            output_dir / f"{doc_filename}-table-{table_counter}.png"
        )
        with element_image_filename.open("wb") as fp:
            img = element.get_image(result.document)
            if img is not None:
                img.save(fp, "PNG")

    if isinstance(element, PictureItem):
        picture_counter += 1
        element_image_filename = (
            output_dir / f"{doc_filename}-picture-{picture_counter}.png"
        )
        with element_image_filename.open("wb") as fp:
            img = element.get_image(result.document)
            if img is not None:
                img.save(fp, "PNG")
"""

# print(json.dumps(result.document.export_to_dict()))



