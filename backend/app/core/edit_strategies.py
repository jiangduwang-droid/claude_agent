import re
import difflib
from dataclasses import dataclass


@dataclass
class EditResult:
    success: bool
    content: str = ""
    strategy: str = ""
    error: str = ""


def _exact(text: str, old: str, new: str) -> str | None:
    if old in text:
        return text.replace(old, new, 1)
    return None


def _trim(text: str, old: str, new: str) -> str | None:
    old_trimmed = old.strip()
    if old_trimmed and old_trimmed in text:
        return text.replace(old_trimmed, new, 1)
    return None


def _normalize_whitespace(text: str, old: str, new: str) -> str | None:
    old_norm = re.sub(r"\s+", " ", old).strip()
    if old_norm:
        pattern = re.escape(old_norm)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[:match.start()] + new + text[match.end():]
    return None


def _ignore_leading_ws(text: str, old: str, new: str) -> str | None:
    old_lines = old.split("\n")
    text_lines = text.split("\n")
    for i in range(len(text_lines) - len(old_lines) + 1):
        match = True
        for j, old_line in enumerate(old_lines):
            if text_lines[i + j].strip() != old_line.strip():
                match = False
                break
        if match:
            text_lines[i:i + len(old_lines)] = new.split("\n")
            return "\n".join(text_lines)
    return None


def _block_match(text: str, old: str, new: str) -> str | None:
    old_blocks = [b.strip() for b in old.split("\n\n") if b.strip()]
    text_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    for i in range(len(text_blocks) - len(old_blocks) + 1):
        if all(text_blocks[i + j] == old_blocks[j] for j in range(len(old_blocks))):
            text_blocks[i:i + len(old_blocks)] = new.split("\n\n")
            return "\n\n".join(text_blocks)
    return None


def _line_match(text: str, old: str, new: str) -> str | None:
    old_lines = old.strip().split("\n")
    if not old_lines:
        return None
    first = old_lines[0].strip()
    last = old_lines[-1].strip()
    text_lines = text.split("\n")
    start = None
    end = None
    for i, line in enumerate(text_lines):
        if start is None and line.strip() == first:
            start = i
        if start is not None and line.strip() == last:
            end = i
            break
    if start is not None and end is not None and end >= start:
        text_lines[start:end + 1] = new.split("\n")
        return "\n".join(text_lines)
    return None


def _fuzzy_match(text: str, old: str, new: str, threshold: float = 0.7) -> str | None:
    ratio = difflib.SequenceMatcher(None, old, text).ratio()
    if ratio < threshold:
        return None
    matcher = difflib.SequenceMatcher(None, old, text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" or tag == "delete":
            return text[:j1] + new + text[j2:]
    return None


def _fuzzy_original(text: str, old: str, new: str, threshold: float = 0.7) -> str | None:
    ratio = difflib.SequenceMatcher(None, old, text).ratio()
    if ratio < threshold:
        return None
    matcher = difflib.SequenceMatcher(None, old, text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            return text[:j1] + new + text[j2:]
    return None


def _wildcard(text: str, old: str, new: str) -> str | None:
    if "..." not in old:
        return None
    parts = old.split("...")
    if len(parts) != 2:
        return None
    start_pat, end_pat = parts[0].strip(), parts[1].strip()
    start_idx = text.find(start_pat)
    if start_idx == -1:
        return None
    end_idx = text.find(end_pat, start_idx + len(start_pat))
    if end_idx == -1:
        return None
    return text[:start_idx] + new + text[end_idx + len(end_pat):]


STRATEGIES = [
    ("exact", _exact),
    ("trim", _trim),
    ("normalize_whitespace", _normalize_whitespace),
    ("ignore_leading_ws", _ignore_leading_ws),
    ("block_match", _block_match),
    ("line_match", _line_match),
    ("fuzzy_normalize", lambda t, o, n: _fuzzy_match(re.sub(r"\s+", " ", t), re.sub(r"\s+", " ", o), n)),
    ("fuzzy_original", _fuzzy_original),
    ("wildcard", _wildcard),
]


def apply_edit(text: str, old: str, new: str) -> EditResult:
    if not text:
        return EditResult(False, error="Empty file content")

    for name, strategy in STRATEGIES:
        try:
            result = strategy(text, old, new)
            if result is not None:
                return EditResult(True, content=result, strategy=name)
        except Exception:
            continue

    return EditResult(False, error="All 9 strategies failed to match")


def apply_patch(content: str, diff: str) -> str:
    lines = content.split("\n")
    patch_lines = diff.split("\n")
    result_lines = list(lines)
    offset = 0

    i = 0
    while i < len(patch_lines):
        line = patch_lines[i]
        match = re.match(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", line)
        if match:
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1

            i += 1
            hunk_lines = []
            while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
                hunk_lines.append(patch_lines[i])
                i += 1

            replaced = []
            for hline in hunk_lines:
                if hline.startswith("-"):
                    continue
                elif hline.startswith("+"):
                    replaced.append(hline[1:])
                elif hline.startswith(" "):
                    replaced.append(hline[1:])
                elif hline == "":
                    replaced.append("")

            actual_start = old_start - 1 + offset
            result_lines[actual_start:actual_start + old_count] = replaced
            offset += len(replaced) - old_count
        else:
            i += 1

    return "\n".join(result_lines)
