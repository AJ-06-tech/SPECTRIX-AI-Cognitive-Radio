from pathlib import Path
import re

path = Path("README.md")
text = path.read_text(encoding="utf-8")

lines = text.splitlines(keepends=True)
fixed = []
inside_fence = False

for line in lines:
    if line.lstrip().startswith("```"):
        inside_fence = not inside_fence
        fixed.append(line)
        continue

    if inside_fence:
        fixed.append(line)
        continue

    parts = re.split(r"(`+.*?`+)", line)

    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\\([#*_\[\]<>.!&|+-])", r"\1", parts[i])

    fixed.append("".join(parts))

path.write_text("".join(fixed), encoding="utf-8")
print("README formatting correction completed.")
