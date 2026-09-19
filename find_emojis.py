import re
from pathlib import Path

for file in ["app.py", "src/models.py", "train_models.py"]:
    content = Path(file).read_text()
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    matches = emoji_pattern.findall(content)
    if matches:
        print(f"{file}: {set(matches)}")
