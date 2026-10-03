"""Build the static GitHub Pages site into docs/ from the page sources.

  python build_site.py

routing diagram : presentation.html        -> docs/routing.html
3D journey      : journey3d.fragment.html  -> docs/journey3d.html
floorplan flow  : finance.fragment.html    -> docs/finance.html
shell (1 page)  : site_shell.html          -> docs/index.html
Also refreshes static/ (served by the FastAPI demo).
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).parent
HEAD = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body style="margin:0">\n'
TAIL = '\n</body></html>\n'
PAGES = {"routing.html": "presentation.html", "journey3d.html": "journey3d.fragment.html", "finance.html": "finance.fragment.html"}

if __name__ == "__main__":
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    for out, src in PAGES.items():
        html = HEAD + (ROOT / src).read_text(encoding="utf-8") + TAIL
        (docs / out).write_text(html, encoding="utf-8")
        if out != "routing.html":
            (ROOT / "static" / out).write_text(html, encoding="utf-8")
    shutil.copyfile(ROOT / "site_shell.html", docs / "index.html")
    (docs / ".nojekyll").write_text("")
    print("built docs/:", sorted(p.name for p in docs.iterdir()))
