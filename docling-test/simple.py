import json
from docling.document_converter import DocumentConverter

################################################################################
# Very basic vanilla test for PDF extractions
################################################################################
source = "/Users/dchang/workspace/askem/pdfs/SIDARTHE_Giordano2020.pdf"

converter = DocumentConverter()
result = converter.convert(source)
print(json.dumps(result.document.export_to_dict()))
