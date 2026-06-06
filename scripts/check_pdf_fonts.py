import json
import sys
from pathlib import Path
try:
    from PyPDF2 import PdfReader
    from PyPDF2.generic import IndirectObject
except Exception as e:
    print('ERROR: PyPDF2 not installed:', e)
    sys.exit(2)
pdf = Path('output/analysis_report.pdf')
if not pdf.exists():
    print('ERROR: PDF not found at', pdf)
    sys.exit(2)
reader = PdfReader(str(pdf))
fonts = set()
embedded = {}
for i,page in enumerate(reader.pages):
    res = page.get('/Resources') or page.get('Resources')
    if not res:
        continue
    fnts = res.get('/Font') or res.get('Font')
    if not fnts:
        continue
    # resolve IndirectObject if necessary
    try:
        if isinstance(fnts, IndirectObject):
            fnts = fnts.get_object()
    except Exception:
        pass
    try:
        items = fnts.items()
    except Exception:
        items = list(fnts.items())
    for k,v in items:
        try:
            if isinstance(v, IndirectObject):
                v = v.get_object()
        except Exception:
            pass
        try:
            base = v.get('/BaseFont') or v.get('BaseFont')
            base = str(base)
        except Exception:
            base = str(v)
        fonts.add(base)
        fd = None
        try:
            fd = v.get('/FontDescriptor') or v.get('FontDescriptor')
        except Exception:
            # maybe indirect
            try:
                fd = v.get_object().get('/FontDescriptor')
            except Exception:
                fd = None
        has = False
        if fd:
            try:
                if isinstance(fd, IndirectObject):
                    fd = fd.get_object()
            except Exception:
                pass
            for key in ('/FontFile','/FontFile2','/FontFile3'):
                try:
                    if fd.get(key):
                        has = True
                        break
                except Exception:
                    pass
        embedded[base] = embedded.get(base, False) or has
print(json.dumps({'fonts': sorted(list(fonts)), 'embedded': embedded}, indent=2))
