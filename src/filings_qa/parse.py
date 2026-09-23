"""Filing HTML to plain text, and plain text to the filing's "Item" sections."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning
from bs4.element import PreformattedString

MIN_SECTION_WORDS = 200  # the item sequence starts at a heading followed by at least this many words

_SKIP = frozenset({"script", "style", "noscript", "head", "title", "meta", "link", "template", "ix:header"})
_BLOCK = frozenset(
    "address article aside blockquote body caption center dd div dl dt figcaption figure footer form h1 h2 h3 h4 h5"
    " h6 header hr html li main nav ol p pre section ul".split()
)
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)
_SOURCE_BREAKS = re.compile(r"[\r\n]+")
_SPACES = re.compile(r"[^\S\t\n]+")
_TABS = re.compile(r" *\t[\t ]*")
_PAGE_NUMBER = re.compile(r"\d{1,3}")
_CURRENCY = frozenset("$€£¥")
_CLOSING = re.compile(r"[)%]+")

_ITEM = re.compile(r"^\s*Item\s+(\d{1,2}[A-C]?)(?![0-9A-Za-z])\.?\s*[—–-]?\s*(.{0,80})", re.I)
_NOT_HEADING = re.compile(r"\s+[a-z]|\s*[,;)]")  # "Item 1A of our 10-K ...": a cross-reference, not a heading
_PART = re.compile(r"^\s*PART\s+(IV|I{1,3})\b", re.I)
_CONTENTS_ROW = re.compile(r"\t\d{1,3}(?:\s*[-–]\s*\d{1,3})?$")  # a row ending in a page number (or range) cell
_PART_NUMBER = {"I": 1, "II": 2, "III": 3, "IV": 4}
_PART_NAME = {v: k for k, v in _PART_NUMBER.items()}
_ITEM_ORDER = re.compile(r"(\d+)([A-C]?)")


@dataclass(frozen=True)
class Section:
    item: str  # "1A", "7", "2"; "II-1" when a later part reuses an item number (10-Q Part II); "" outside any item
    title: str
    text: str


def html_to_text(html: str | bytes) -> str:
    """Plain text of a filing: one line per paragraph or table row, table cells separated by tabs.

    Scripts, styles, hidden elements (such as the inline XBRL header) and bare page-number lines are dropped, and
    runs of spaces are collapsed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)  # inline XBRL is XHTML; the HTML parser is intended
        soup = BeautifulSoup(html, "lxml")
    parts: list[str] = []
    _emit(soup, parts)
    return _normalize("".join(parts))


def _is_hidden(tag: Tag) -> bool:
    style = tag.get("style")
    return isinstance(style, str) and _HIDDEN.search(style) is not None


def _emit(root: Tag, out: list[str]) -> None:
    """Append the text under ``root`` to ``out``, with newlines around block elements (iterative, so deeply nested
    documents cannot hit the recursion limit)."""
    stack: list[object] = list(reversed(root.contents))
    while stack:
        node = stack.pop()
        if isinstance(node, Tag):
            if node.name in _SKIP or _is_hidden(node):
                continue
            if node.name == "br":
                out.append("\n")
            elif node.name == "pre":
                out.append("\n" + node.get_text() + "\n")
            elif node.name == "table":
                out.append("\n" + _table_text(node) + "\n")
            else:
                if node.name in _BLOCK:
                    out.append("\n")
                    stack.append("\n")  # emitted after the element's children
                stack.extend(reversed(node.contents))
        elif isinstance(node, NavigableString):
            if not isinstance(node, PreformattedString):  # skips comments, doctype, etc.
                out.append(_SOURCE_BREAKS.sub(" ", node))  # a line break in the source is just a space
        else:
            out.append(node)  # the "\n" pushed after a block element's children


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table or _is_hidden(tr):
            continue  # rows of nested tables are handled inside their cell
        cells = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            if _is_hidden(cell):
                continue
            parts: list[str] = []
            _emit(cell, parts)
            cells.append(" ".join("".join(parts).split()))
        cells = _merge_cells(cells)
        if cells:
            rows.append("\t".join(cells))
    return "\n".join(rows)


def _merge_cells(cells: list[str]) -> list[str]:
    """Drop empty cells; glue a lone "$" to the next cell and a lone ")" or "%" to the previous one."""
    merged: list[str] = []
    carry = ""
    for cell in cells:
        if not cell:
            continue
        if cell in _CURRENCY:
            carry += cell
        elif merged and not carry and _CLOSING.fullmatch(cell):
            merged[-1] += cell
        else:
            merged.append(carry + cell)
            carry = ""
    if carry:
        merged.append(carry)
    return merged


def _normalize(raw: str) -> str:
    lines = []
    for line in raw.replace("\u200b", "").replace("\ufeff", "").split("\n"):
        line = _TABS.sub("\t", _SPACES.sub(" ", line)).strip(" \t")
        if line and not _PAGE_NUMBER.fullmatch(line):
            lines.append(line)
    return "\n".join(lines)


@dataclass
class _Heading:
    line: int
    part: int  # 1 for Part I, ...; 0 before the body's first part heading
    item: str
    title: str
    weight: int = 0  # words from this heading to the next heading of another item

    @property
    def order(self) -> tuple[int, int, str]:
        m = _ITEM_ORDER.fullmatch(self.item)
        assert m is not None
        return self.part, int(m.group(1)), m.group(2)


def _clean_title(raw: str) -> str:
    title = " ".join(raw.split()).strip(" .:-—–")
    return re.split(r"(?<=[a-z])\. ", title, maxsplit=1)[0]  # a heading run into its first sentence


def _is_contents_row(line: str) -> bool:
    return _CONTENTS_ROW.search(line) is not None


def _heads_contents_rows(block: list[str]) -> bool:
    """True when ``block`` (the lines up to the next item line) is mostly short rows that end in a page number: the
    item line above it is a group heading of a contents page (one filer lists each statement and note this way)."""
    rows = sum(1 for line in block if _is_contents_row(line) and line.count("\t") <= 2)
    return rows >= 2 and rows >= 0.8 * len(block)


def _find_headings(lines: list[str]) -> tuple[list[_Heading], set[int], dict[str, int]]:
    """Candidate item headings; the lines that only help navigation (part headings, contents-page rows); and for
    each item number, the first part (1 for Part I) that uses it anywhere in the document."""
    item_lines = [
        (i, m) for i, line in enumerate(lines) if (m := _ITEM.match(line)) and not _NOT_HEADING.match(line, m.end(1))
    ]
    next_item_line = {i: j for (i, _), (j, _) in zip(item_lines, item_lines[1:], strict=False)}
    matches = dict(item_lines)
    part = 0  # the part of the body text, set by part headings in the body
    listed_part = 0  # the part as the contents page lists it
    headings: list[_Heading] = []
    navigation: set[int] = set()
    first_part: dict[str, int] = {}
    for i, line in enumerate(lines):
        if len(line) <= 80 and (m := _PART.match(line)):
            navigation.add(i)
            listed_part = _PART_NUMBER[m.group(1).upper()]
            if not (i + 1 < len(lines) and _is_contents_row(lines[i + 1])):
                part = listed_part  # a part row of the contents page does not start that part in the body
            continue
        m = matches.get(i)
        if m is None:
            continue
        item = m.group(1).upper()
        if _is_contents_row(line) or _heads_contents_rows(lines[i + 1 : next_item_line.get(i, len(lines))]):
            navigation.add(i)  # a contents-page row: it ends in a page-number cell or heads rows that do
            if listed_part:
                first_part[item] = min(first_part.get(item, listed_part), listed_part)
            continue
        if part:
            first_part[item] = min(first_part.get(item, part), part)
        title = _clean_title(m.group(2))
        if not title and i + 1 < len(lines) and len(lines[i + 1]) <= 100 and not _ITEM.match(lines[i + 1]):
            title = _clean_title(lines[i + 1])
        headings.append(_Heading(i, part, item, title))
    return headings, navigation, first_part


def _best_sequence(headings: list[_Heading]) -> list[_Heading]:
    """The headings, in document order and in increasing (part, item) order, that start the most text.

    A contents page lists the same items as the body but with almost no text after each line, so the body headings
    win; a page header that repeats an item's heading starts less text than the real heading, so it loses too.
    """
    n = len(headings)
    best = [h.weight + 1 for h in headings]
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if headings[j].order < headings[i].order and best[j] + headings[i].weight + 1 > best[i]:
                best[i] = best[j] + headings[i].weight + 1
                prev[i] = j
    i = max(range(n), key=best.__getitem__)
    chain = []
    while i != -1:
        chain.append(headings[i])
        i = prev[i]
    return chain[::-1]


def split_items(text: str) -> list[Section]:
    """Split filing text at its "Item N." headings.

    Candidate headings are lines starting with ``Item <number>[A-C]``, except contents-page rows (they end in a
    page-number cell, or head a group of short rows that do). Other contents-page lines and repeated page headers are
    filtered out by keeping the in-order sequence of headings that starts the most text, and the sequence starts at
    the first heading followed by at least ``MIN_SECTION_WORDS`` words. Text before it becomes a section with item ""
    (contents-page lines removed). When a later part reuses an item number of an earlier part (10-Q Part II Items 1
    to 4), the label gets the part as a prefix, e.g. "II-1". Without any usable heading (some filers put item
    headings only on the contents page), the whole text is one section with item "".
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    headings, navigation, first_part = _find_headings(lines)
    words_before = [0]
    for line in lines:
        words_before.append(words_before[-1] + len(line.split()))
    for k, h in enumerate(headings):
        nxt = k + 1
        while nxt < len(headings) and (headings[nxt].part, headings[nxt].item) == (h.part, h.item):
            nxt += 1
        end = headings[nxt].line if nxt < len(headings) else len(lines)
        h.weight = words_before[end] - words_before[h.line]

    chain = _best_sequence(headings) if headings else []
    first = next((k for k, h in enumerate(chain) if h.weight >= MIN_SECTION_WORDS), None)
    if first is None:
        return [Section("", "", "\n".join(lines))]
    chain = chain[first:]

    sections = []
    skip = navigation | {h.line for h in headings}
    front = [line for i, line in enumerate(lines[: chain[0].line]) if i not in skip]
    if front:
        sections.append(Section("", "Front matter", "\n".join(front)))
    for k, h in enumerate(chain):
        label = f"{_PART_NAME[h.part]}-{h.item}" if h.part > first_part.get(h.item, h.part) else h.item
        end = chain[k + 1].line if k + 1 < len(chain) else len(lines)
        sections.append(Section(label, h.title, "\n".join(lines[h.line : end])))
    return sections
