import re
from pathlib import Path

content = Path("app.py").read_text()

replacements = {
    'page_icon="🛒",': '',
    '<h2>✨ Daraz ABSA</h2>': '<h2>Daraz ABSA</h2>',
    '"🔍 Review Analyzer"': '"Review Analyzer"',
    '"📈 Benchmarks & Data Insights"': '"Benchmarks & Data Insights"',
    '"💡 Quick Test Presets:"': '"Quick Test Presets:"',
    '"🚀 Analyze Review"': '"Analyze Review"',
    's_ico = "😊" if s_lbl == "Positive" else ("😡" if s_lbl == "Negative" else "😐")': 's_ico = ""',
    'icon = "⚠️"': 'icon = "[Warning]"',
    'icon = "✅"': 'icon = "[Pass]"',
    'icon = "😡"': 'icon = "[Fail]"',
}

for old, new in replacements.items():
    content = content.replace(old, new)

# And in models.py there are emojis too! Let's check models.py
Path("app.py").write_text(content)
