import easyocr
import warnings
import sys
warnings.filterwarnings('ignore')
print('Downloading models / Loading...', flush=True)
reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
print('Reading page 1...', flush=True)
p1 = reader.readtext('research/page1.png', detail=0)
print('PAGE 1 TEXT: ' + ' '.join(p1), flush=True)
print('Reading page 2...', flush=True)
p2 = reader.readtext('research/page2.png', detail=0)
print('PAGE 2 TEXT: ' + ' '.join(p2), flush=True)
