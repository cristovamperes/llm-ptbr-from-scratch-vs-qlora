import json
from pathlib import Path

data = json.loads(Path('analysis/samples_v5_scaleplus.json').read_text(encoding='utf-8'))
for item in data['v5_brwac_transformer_fixed']:
    print('\nSEED', item['seed'])
    print(item['output'])
