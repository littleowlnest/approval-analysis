from pathlib import Path
import sys

p = Path(sys.argv[1]) if len(sys.argv)>1 else Path('input/docs/ANIS NABILA BINTI ZINALIBIN.pdf')
if not p.exists():
    print('Missing file', p)
    raise SystemExit(2)

b = p.read_bytes()
# heuristics
s = b[:20000]
try:
    text = s.decode('latin1')
except Exception:
    text = ''.join(chr(x) if 32<=x<127 else '.' for x in s[:1000])

keywords = ['Score','score','CTOS','ctos','CREDIT','Credit','/Font','/Page','BT','ET','Image','XObject']
found = {k: (k.lower() in text.lower()) for k in keywords}
print('File:', p)
for k,v in found.items():
    print(f'{k}:', v)

# print a short ascii sample
print('\n--- ASCII sample ---')
print(text[:1000])

# count 3-digit numbers in first chunk
import re
nums = re.findall(r'\b(\d{3})\b', text)
print('\n3-digit numbers found (sample):', nums[:20])
print('Total 3-digit in sample:', len(nums))
