"""
scripts/rebuild_reports.py
──────────────────────────
Rebuild eval/RESULTS_<tag>.md from the saved eval/results_<tag>.json, WITHOUT
re-running any question: no number changes, only the report text.

Why it exists: the first runs of 5 Nov 2026 labelled a Windows 11 PC "Windows 10"
(Python 3.11 reports both as release "10"). The JSON is corrected only when the run
was made on THIS machine (same CPU), so a report is never relabelled with another
computer's OS.

    python -m scripts.rebuild_reports
"""

import json

from scripts.evaluate import EVAL, QUESTION_FILES, _cpu_name, _os_name, write_markdown


def main() -> None:
    for path in sorted(EVAL.glob("results_*.json")):
        tag = path.stem.removeprefix("results_")
        result = json.loads(path.read_text(encoding="utf-8"))
        machine = result.get("settings", {}).get("machine")
        if not machine:
            print(f"skip {path.name}: older run without machine info (report kept as is)")
            continue
        if machine["cpu"] == _cpu_name() and machine["os"] != _os_name():
            print(f"{path.name}: os {machine['os']!r} -> {_os_name()!r}")
            machine["os"] = _os_name()
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        # The split is the longest question-set name the tag starts with ("dev_classes_drugclasses" -> dev_classes)
        split = max((s for s in QUESTION_FILES if tag.startswith(s)), key=len)
        write_markdown(result, split, tag)
        print(f"rebuilt RESULTS_{tag}.md (split {split})")


if __name__ == "__main__":
    main()
