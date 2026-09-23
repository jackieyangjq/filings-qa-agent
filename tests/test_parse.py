from filings_qa.parse import html_to_text, split_items


def _filler(word: str, n: int) -> str:
    """A paragraph of ``n`` words."""
    return " ".join(f"{word}{i}" for i in range(n))


def test_html_to_text_drops_scripts_styles_and_hidden_xbrl(sample_html):
    text = html_to_text(sample_html)
    assert "trackingPixel" not in text
    assert "font-family" not in text
    assert "HIDDENXBRLFACT" not in text  # inline XBRL header inside display:none
    assert "acme-20240629" not in text  # <title>
    assert "Acme Widgets, Inc." in text


def test_html_to_text_lines_and_whitespace(sample_html):
    lines = html_to_text(sample_html).splitlines()
    assert "Item 1. Financial Statements" in lines  # non-breaking spaces collapsed
    assert "UNITED STATES" in lines and "SECURITIES AND EXCHANGE COMMISSION" in lines  # <br/> starts a new line
    # a line break inside a paragraph of the source is only a space
    assert any(line.startswith("This report contains forward-looking statements") and line.endswith("statement.")
               for line in lines)
    assert not any(line.isdigit() for line in lines)  # page numbers dropped


def test_table_rows_are_tab_separated(sample_html):
    lines = html_to_text(sample_html).splitlines()
    assert "Net revenue\t$12,345\t$11,000" in lines  # "$" joined to the amount
    assert "Cost of revenue\t(6,789)\t(6,100)" in lines  # ")" joined to the amount
    assert "Gross margin percentage\t45.0%\t44.5%" in lines
    assert "Item 1A.\tRisk Factors\t8" in lines  # contents page is a table too


def test_split_items_finds_body_sections_and_ignores_contents_page(sample_html):
    sections = split_items(html_to_text(sample_html))
    assert [s.item for s in sections] == ["", "1", "2", "II-1", "1A", "7"]
    by_item = {s.item: s for s in sections}

    risk = by_item["1A"]
    assert risk.title == "Risk Factors"
    assert risk.text.startswith("Item 1A. Risk Factors")
    assert "specialty steel" in risk.text and "Supplementary Information" not in risk.text

    supplementary = by_item["7"]
    assert supplementary.title == "Supplementary Information"
    assert "Backlog." in supplementary.text and len(supplementary.text.split()) > 200

    # the cross-reference "Item 1A of the Company's Annual Report ..." does not cut MD&A short
    assert "Liquidity and Capital Resources" in by_item["2"].text
    # Part II reuses item number 1, so its label names the part
    assert by_item["II-1"].title == "Legal Proceedings"
    assert "legal proceedings and claims" in by_item["II-1"].text

    front = by_item[""]
    assert "FORM 10-Q" in front.text and "Forward-Looking Statements" in front.text
    assert "Risk Factors" not in front.text  # contents-page lines removed


def test_split_items_without_headings_returns_whole_text():
    text = "Annual report\n" + _filler("w", 300)
    assert [(s.item, s.text) for s in split_items(text)] == [("", text)]
    assert split_items("") == []


def test_split_items_keeps_short_body_items_after_the_first_real_section():
    text = "\n".join(
        [
            "Item 1. Financial Statements",  # contents page without page numbers
            "Item 2. Management's Discussion",
            "Item 3. Defaults",
            "Item 1. Financial Statements",
            _filler("fin", 250),
            "Item 2. Management's Discussion",
            _filler("mda", 250),
            "Item 3. Defaults Upon Senior Securities",
            "None.",
        ]
    )
    sections = split_items(text)
    assert [s.item for s in sections] == ["1", "2", "3"]
    assert sections[0].text.startswith("Item 1. Financial Statements\nfin0 ")
    assert sections[2].text == "Item 3. Defaults Upon Senior Securities\nNone."


def test_split_items_ignores_repeated_page_headers():
    text = "\n".join(
        [
            "Item 7. Management's Discussion and Analysis",
            _filler("page1_", 300),
            "Item 7. Management's Discussion and Analysis (continued)",  # running header on the next page
            _filler("page2_", 300),
            "Item 8. Financial Statements",
            _filler("fs", 300),
        ]
    )
    sections = split_items(text)
    assert [s.item for s in sections] == ["7", "8"]
    assert "page1_0" in sections[0].text and "page2_0" in sections[0].text


def test_split_items_contents_line_followed_by_long_text_is_not_a_section():
    # the last contents line is followed by a long preface before the body starts
    text = "\n".join(
        [
            "Item 1. Business",
            "Item 16. Form 10-K Summary",
            _filler("preface", 300),
            "Item 1. Business",
            _filler("biz", 400),
            "Item 16. Form 10-K Summary",
            "None.",
        ]
    )
    sections = split_items(text)
    assert [s.item for s in sections] == ["", "1", "16"]
    assert sections[0].text == _filler("preface", 300)


def test_split_items_headings_only_on_the_contents_page():
    # some filers (a large bank in the corpus) list the items with page numbers and never repeat them in the body
    text = "\n".join(
        [
            "FORM 10-Q",
            "Part I – Financial information",
            "Item 1.\tFinancial Statements.\t93",
            "Item 2.\tManagement's Discussion and Analysis.\t3",
            "Part II – Other information",
            "Item 1A.\tRisk Factors.\t201",
            "Item 6.\tExhibits.\t202-203",
            _filler("report", 900),
        ]
    )
    assert [(s.item, s.text) for s in split_items(text)] == [("", text)]


def test_split_items_contents_page_groups_and_closing_items_only():
    # modeled on a large bank's 10-Q: the contents page groups statements and notes under Items 1 and 2 with page
    # numbers, the report itself has no item headings, and only the closing items appear as headings
    text = "\n".join(
        [
            "FORM 10-Q",
            "Part I – Financial information\tPage",
            "Item 1.\tFinancial Statements",
            "Consolidated statements of income\t80",
            "Consolidated balance sheets\t82",
            "Note 1 - Basis of presentation\t85",
            "Item 2.\tManagement's Discussion and Analysis.",
            "Executive Overview\t5",
            "Consolidated Results of Operations\t9",
            "Item 3.\tQuantitative and Qualitative Disclosures About Market Risk.\t179",
            "Part II – Other information",  # a contents-page row: does not start Part II in the body
            "Item 1.\tLegal Proceedings.\t179",
            "Item 1A.\tRisk Factors.\t179",
            _filler("report", 900),
            "Item 3. Quantitative and Qualitative Disclosures About Market Risk.",
            _filler("market", 250),
            "Part II – Other Information",
            "Item 1. Legal Proceedings.",
            _filler("legal", 250),
            "Item 1A. Risk Factors.",
            _filler("risk", 30),
        ]
    )
    sections = split_items(text)
    assert [s.item for s in sections] == ["", "3", "II-1", "1A"]
    assert "report0" in sections[0].text and "Executive Overview\t5" in sections[0].text
    assert "Item 2.\tManagement's Discussion and Analysis." not in sections[0].text


def test_split_items_heading_followed_by_an_index_table_stays_a_heading():
    text = "\n".join(
        [
            "Item 7. Management's Discussion and Analysis",
            _filler("mda", 300),
            "Item 8. Financial Statements and Supplementary Data",
            "Index to Consolidated Financial Statements\tPage",
            "Consolidated Statements of Operations\t28",
            "Consolidated Balance Sheets\t30",
            "Notes to Consolidated Financial Statements\t33",
            _filler("fs", 400),
            "Item 9. Changes in and Disagreements with Accountants",
            "None.",
        ]
    )
    assert [s.item for s in split_items(text)] == ["7", "8", "9"]
