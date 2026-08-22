from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


root = Path(__file__).resolve().parents[1]
qa = root / "artifacts" / "docx_qa_final"
pages = sorted(qa.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
thumbs = []
for i, path in enumerate(pages, 1):
    image = Image.open(path).convert("RGB")
    target_w = 330
    target_h = round(image.height * target_w / image.width)
    thumb = image.resize((target_w, target_h))
    canvas = Image.new("RGB", (target_w + 16, target_h + 42), "#E5E7EB")
    canvas.paste(thumb, (8, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 7), f"Page {i}", fill="#111827")
    thumbs.append(canvas)

cols = 3
rows = (len(thumbs) + cols - 1) // cols
cell_w = max(t.width for t in thumbs)
cell_h = max(t.height for t in thumbs)
sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#9CA3AF")
for i, thumb in enumerate(thumbs):
    sheet.paste(thumb, ((i % cols) * cell_w, (i // cols) * cell_h))
sheet.save(qa / "contact-sheet.png")
print(qa / "contact-sheet.png")
