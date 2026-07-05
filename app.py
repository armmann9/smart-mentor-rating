# ============================================================================
# MENTOR RESUME PARSER + RATING PREDICTOR
# ----------------------------------------------------------------------------
# What this app does, in plain terms:
#   1. User uploads a resume PDF.
#   2. We pull out the raw text (and hidden hyperlinks) from the PDF.
#   3. We split that text into sections (Skills, Experience, Projects, etc.)
#      using regex heading patterns.
#   4. We display everything nicely on screen.
#   5. We also fetch the person's public GitHub repos (if a GitHub link/handle
#      was found) and count total projects (resume + GitHub, de-duplicated).
#   6. Finally we feed a few numeric features (years of experience, skill
#      count, project count, a default feedback score) into a pre-trained
#      XGBoost model to predict a "Mentor Rating" out of 10.
# ============================================================================

import streamlit as st
import pdfplumber
import re
import json
import requests
import pandas as pd
import pickle

st.set_page_config(page_title="Resume Parser", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------------
# All the CSS for the page lives here as one big style block. Nothing in
# here affects logic — it only controls colors, spacing, fonts, etc.
# ----------------------------------------------------------------------------
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.main-title {
    font-size: 2.2rem; font-weight: 800; color: #1e293b;
    text-align: center; margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center; color: #64748b; margin-bottom: 1.5rem; font-size: 1rem;
}
.pipeline {
    display: flex; justify-content: center; align-items: center;
    gap: 0.4rem; flex-wrap: wrap;
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    border-radius: 10px; padding: 0.8rem 1rem;
    color: white; font-size: 0.9rem; margin-bottom: 2rem;
    font-weight: 500;
}
.section-wrap {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.sec-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6366f1;
    margin-bottom: 0.6rem;
    border-bottom: 2px solid #e0e7ff;
    padding-bottom: 0.4rem;
}
.name-text {
    font-size: 1.7rem;
    font-weight: 800;
    color: #1e293b;
    margin: 0;
}
.contact-row {
    display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.4rem;
}
.contact-chip {
    background: #f1f5f9; border: 1px solid #cbd5e1;
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.82rem; color: #334155;
}
.skill-tag {
    display: inline-block;
    background: #ede9fe; color: #4c1d95;
    border-radius: 20px; padding: 3px 13px;
    font-size: 0.82rem; margin: 3px 3px;
    font-weight: 500;
}
.content-text {
    font-size: 0.88rem; color: #374151;
    white-space: pre-wrap; line-height: 1.7;
}
.entry-block {
    border-left: 3px solid #c7d2fe;
    padding-left: 0.8rem;
    margin-bottom: 0.9rem;
}
.entry-title { font-weight: 700; color: #1e293b; font-size: 0.92rem; }
.entry-sub   { color: #6b7280; font-size: 0.82rem; margin-bottom: 0.2rem; }
.entry-body  { font-size: 0.85rem; color: #374151; line-height: 1.6; }
.github-card {
    border: 1.5px solid #d1d5db;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    background: #f8fafc;
    transition: box-shadow 0.2s;
}
.github-card:hover { box-shadow: 0 4px 12px rgba(99,102,241,0.15); border-color: #a5b4fc; }
.github-card-title { font-weight: 700; color: #4f46e5; font-size: 0.92rem; text-decoration: none; }
.github-card-title:hover { text-decoration: underline; }
.github-card-desc { color: #6b7280; font-size: 0.82rem; margin: 0.2rem 0 0.4rem; }
.github-card-meta { display: flex; gap: 0.8rem; flex-wrap: wrap; font-size: 0.78rem; color: #9ca3af; }
.lang-dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:#6366f1; margin-right:3px; }
.gh-divider { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#10b981; margin: 0.8rem 0 0.5rem; border-bottom: 2px solid #d1fae5; padding-bottom:0.3rem; }
.proj-stats-bar {
    display: flex; gap: 0; flex-wrap: wrap;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 10px; padding: 0.75rem 1.2rem;
    margin-bottom: 1rem; align-items: center;
}
.proj-stat-box {
    text-align: center; flex: 1; min-width: 70px;
}
.proj-stat-num {
    font-size: 1.6rem; font-weight: 900; color: #ffffff; line-height: 1;
}
.proj-stat-label {
    font-size: 0.68rem; font-weight: 600; color: #c7d2fe;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;
}
.proj-stat-divider {
    width: 1px; background: rgba(255,255,255,0.35);
    align-self: stretch; min-height: 40px; margin: 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Page header + the little "pipeline" banner showing the 4-step process
st.markdown('<div class="main-title">📄 Resume Parser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your resume PDF — we\'ll extract and display every section clearly</div>', unsafe_allow_html=True)
st.markdown("""
<div class="pipeline">
  📤 Upload &nbsp;→&nbsp; 🔍 Extract Text &nbsp;→&nbsp;
  🗂️ Identify Sections &nbsp;→&nbsp; ✅ Display Results
</div>
""", unsafe_allow_html=True)


# ============================================================================
# STEP 1: TEXT EXTRACTION FROM THE PDF
# ============================================================================
def extract_text(file) -> str:
    """
    Reads every page of the uploaded PDF and returns one big string of text.

    Also pulls out any *hyperlinks* embedded in the PDF (not just visible
    text). This matters because many resume templates show a GitHub/LinkedIn
    icon that is clickable, but the actual URL is never printed as visible
    text — so without this step we'd completely miss the person's GitHub
    link if it's "hidden" behind an icon.
    """
    text = ""
    links = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # Grab the normal, visible text on this page
            t = page.extract_text()
            if t:
                text += t + "\n"

            # Many resume templates put the GitHub/LinkedIn icon as a clickable
            # hyperlink with no visible URL text. Pull those link targets too,
            # via pdfplumber's hyperlink annotations, so they aren't missed.
            try:
                for hl in getattr(page, "hyperlinks", []) or []:
                    uri = hl.get("uri")
                    if uri:
                        links.append(uri)
            except Exception:
                pass

            # Fallback: some PDFs / older pdfplumber versions only expose link
            # URIs via raw annotations rather than page.hyperlinks
            try:
                for annot in page.annots or []:
                    uri = (annot.get("data", {}) or {}).get("A", {}).get("URI")
                    if uri:
                        if isinstance(uri, bytes):
                            uri = uri.decode("utf-8", errors="ignore")
                        links.append(uri)
            except Exception:
                pass

    # Tack any discovered hyperlink URLs onto the end of the text, so the
    # same regex-based detection used everywhere else (for GitHub/LinkedIn)
    # can find them just like it would find visible text.
    if links:
        text += "\n" + "\n".join(links)

    return text.strip()


# ============================================================================
# STEP 2: SECTION SPLITTING
# ============================================================================
# Each resume section (Skills, Experience, Projects, ...) is recognized by
# matching a line against one of these regex patterns. If a line looks like
# a heading (e.g. "Work Experience", "Skills", "Education"), everything
# after it (until the next heading) gets filed under that section.
HEADINGS = {
    "SUMMARY":        r"(professional summary|summary|objective|about me|profile|career objective)",
    "SKILLS":         r"(technical skills?|skills?|core competencies|technologies|tools)",
    "EDUCATION":      r"(education|academic|qualifications)",
    "EXPERIENCE":     r"(work experience|professional experience|employment history|experience)",
    "INTERNSHIP":     r"(internship|intern experience|training experience)",
    "PROJECTS":       r"(projects?|personal projects?|academic projects?|key projects?)",
    "CERTIFICATIONS": r"(certifications?|licenses?|courses?|credentials?)",
    "ACHIEVEMENTS":   r"(achievements?|awards?|honors?|accomplishments?|hackathon)",
    "LANGUAGES":      r"(languages?)",
}

def parse_resume(text: str) -> dict:
    """
    Takes the raw resume text and returns a dictionary like:
        {
          "SUMMARY": [...lines...],
          "SKILLS": [...lines...],
          ...
          "_name": "detected name",
          "_contact": {"Email": [...], "Phone": [...], "LinkedIn": [...], "GitHub": [...]},
          "_skill_tags": [...individual skill words...],
        }
    """
    lines = text.splitlines()
    result = {k: [] for k in HEADINGS}
    result["_name"] = ""
    result["_contact"] = {"Email": [], "Phone": [], "LinkedIn": [], "GitHub": []}

    # --- Contact info extraction (works on the whole text, not line-by-line) ---
    result["_contact"]["Email"]    = list(set(re.findall(r"[\w.\-+]+@[\w.\-]+\.\w{2,}", text)))
    result["_contact"]["Phone"]    = list(set(re.findall(r"(?<!\d)(\+?[\d][\d\s\-().]{7,15}\d)(?!\d)", text)))[:2]
    result["_contact"]["LinkedIn"] = list(set(re.findall(r"linkedin\.com/in/[\w\-]+", text, re.I)))
    result["_contact"]["GitHub"]   = list(set(re.findall(r"github\.com/[\w\-]+", text, re.I)))

    # --- Name detection: heuristic guess ---
    # We assume the person's name is one of the first short lines that
    # doesn't look like an email/phone/URL and has 2-6 words in it.
    for line in lines:
        l = line.strip()
        if l and not re.search(r"[@|/\\]|\d{5,}|http|linkedin|github", l, re.I) and 1 < len(l.split()) <= 6 and len(l) < 50:
            result["_name"] = l
            break

    # --- Walk through every line and bucket it under the current section ---
    # "current" tracks which section heading we're currently under. Every
    # non-heading line gets appended to that section's list. Blank lines are
    # kept (as "") so we can later split entries apart (see lines_to_entries).
    current = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                result[current].append("")
            continue
        matched = False
        for section, pattern in HEADINGS.items():
            if re.fullmatch(pattern, stripped, re.IGNORECASE) or \
               (len(stripped) < 40 and re.match(r"^" + pattern + r"[:\s]*$", stripped, re.IGNORECASE)):
                current = section
                matched = True
                break
        if not matched and current:
            result[current].append(stripped)

    # --- Turn the raw "Skills" text into individual tag-like skill words ---
    # e.g. "Python, SQL | Excel" -> ["Python", "SQL", "Excel"]
    raw_skill = " ".join(result.get("SKILLS", []))
    result["_skill_tags"] = [s.strip() for s in re.split(r"[,|•·/]", raw_skill) if 1 < len(s.strip()) < 40]

    return result


def section_box(title: str, content_html: str):
    """Small helper: wraps content in the standard white 'card' box style."""
    st.markdown(f"""
    <div class="section-wrap">
      <div class="sec-title">{title}</div>
      {content_html}
    </div>""", unsafe_allow_html=True)


# ============================================================================
# GITHUB REPO FETCHER
# ============================================================================
def fetch_github_projects(username: str) -> list:
    """
    Calls GitHub's public API to list a user's public repositories.
    Forks are excluded (we only want original projects the person made).
    Note: GitHub allows 60 unauthenticated requests/hour per IP — if you
    hit that limit, this will just return an empty list.
    """
    try:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            repos = resp.json()
            return [
                {
                    "name": r["name"],
                    "description": r.get("description") or "",
                    "language": r.get("language") or "",
                    "stars": r.get("stargazers_count", 0),
                    "url": r.get("html_url", ""),
                    "topics": r.get("topics", []),
                }
                for r in repos if not r.get("fork", False)
            ]
        else:
            return []
    except Exception:
        return []


# ============================================================================
# EXPERIENCE-YEARS CALCULATOR (used by the mentor rating model)
# ============================================================================
def extract_experience_years(text: str) -> tuple:
    """
    Scans the Experience/Internship text for actual date ranges — things
    like "Jan 2021 - Present", "2019-2022", "06/2020 - 08/2023" — and works
    out how many years of experience they represent.

    How it works:
      1. Find every "<start date> - <end date>" pattern in the text.
      2. Convert each date to just its year (ignoring month/day precision).
      3. Take the EARLIEST start year and the LATEST end year found across
         all jobs/internships, and the difference between them is the
         total years of experience.
    "Present"/"Current"/"Now" as an end date counts as this year.

    Returns: (years, detail_string)
      - years is an integer (minimum 1), or None if no dates were found at all.
      - detail_string is a human-readable note of which date ranges were
        used, so the number shown to the user is fully explainable instead
        of a black box.
    """
    import datetime
    current_year = datetime.date.today().year

    # Matches month names (with common abbreviations) e.g. "Jan", "January"
    month = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    # A single date "token" can look like: "Jan 2021", "06/2021", or just "2021"
    date_tok = rf"(?:{month}\.?\s+\d{{4}}|\d{{1,2}}[/\-]\d{{4}}|\d{{4}})"

    # A full range looks like: <date_tok> - <date_tok or "present/current/...">
    # Using named groups (?P<start>...) / (?P<end>...) instead of numbered
    # groups avoids bugs from miscounting nested parentheses.
    range_pattern = re.compile(
        rf"(?P<start>{date_tok})\s*(?:-|–|—|to)\s*(?P<end>present|current|till date|now|{date_tok})",
        re.I
    )

    def token_to_year(tok):
        """Pulls just the 4-digit year out of a matched date token."""
        m = re.search(r"\d{4}", tok)
        return int(m.group()) if m else None

    spans = []
    for m in range_pattern.finditer(text):
        start_year = token_to_year(m.group("start"))
        end_raw = m.group("end")
        if re.search(r"present|current|till date|now", end_raw, re.I):
            end_year = current_year
        else:
            end_year = token_to_year(end_raw)
        # Sanity check: end must be >= start, and the start year shouldn't
        # be some absurd number of decades ago (guards against false matches
        # like phone numbers or IDs accidentally looking like a year range).
        if start_year and end_year and end_year >= start_year and (current_year - start_year) < 60:
            spans.append((start_year, end_year))

    if not spans:
        return None, "no date ranges detected"

    earliest = min(s[0] for s in spans)
    latest = max(s[1] for s in spans)
    years = max(1, latest - earliest)
    detail = ", ".join(f"{s}-{'Present' if e == current_year else e}" for s, e in spans)
    return years, detail


def lines_to_entries(lines):
    """
    Groups a flat list of lines into separate "entries" (e.g. separate jobs,
    separate degrees) by splitting wherever there's a blank line ("") acting
    as a separator between entries.
    """
    entries, current_entry = [], []
    for l in lines:
        if l == "":
            if current_entry:
                entries.append(current_entry)
                current_entry = []
        else:
            current_entry.append(l)
    if current_entry:
        entries.append(current_entry)
    return entries


# ============================================================================
# MAIN APP FLOW — everything below only runs once a PDF has been uploaded
# ============================================================================
uploaded = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded:
    # ---- Extract + parse ----
    with st.spinner("Reading PDF…"):
        raw_text = extract_text(uploaded)
    if not raw_text:
        st.error("No text found. The PDF may be image/scanned. Try a text-based PDF.")
        st.stop()

    with st.spinner("Parsing sections…"):
        data = parse_resume(raw_text)

    st.success(f"✅ Parsed **{uploaded.name}** successfully!")
    st.markdown("---")

    # ── NAME ──────────────────────────────────────────────────────────────────
    name = data["_name"]
    section_box("👤 Name", f'<p class="name-text">{name or "Not detected"}</p>')

    # ── CONTACT ───────────────────────────────────────────────────────────────
    cp = data["_contact"]
    contact_chips = ""
    for label, vals in cp.items():
        for v in vals:
            contact_chips += f'<span class="contact-chip">📌 {label}: {v}</span>'
    if contact_chips:
        section_box("📬 Contact", f'<div class="contact-row">{contact_chips}</div>')

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    summary = [l for l in data.get("SUMMARY", []) if l]
    if summary:
        section_box("📝 Professional Summary", f'<p class="content-text">{chr(10).join(summary)}</p>')

    # ── Pre-calculate project counts BEFORE columns ───────────────────────────
    # NOTE: This block duplicates the logic that also runs later inside
    # col2 (see "PROJECTS (resume + GitHub)" further down). It exists so the
    # "Total Unique Projects" number is available for the stats bar that gets
    # rendered before the two-column layout is drawn. Both blocks compute the
    # same numbers; if you ever change one, change the other to match.
    import difflib
    proj_raw_pre   = [l for l in data.get("PROJECTS", []) if l]
    gh_links_pre   = data["_contact"].get("GitHub", [])
    gh_repos_pre   = []
    gh_username_pre = None
    if gh_links_pre:
        # Pull the username out of a link like "github.com/johndoe" -> "johndoe"
        match_pre = re.search(r"github\.com/([^/\s]+)", gh_links_pre[0], re.I)
        if match_pre:
            # .strip() removes trailing punctuation that can get glued onto
            # the username, e.g. "(github.com/johndoe)" or "johndoe,"
            gh_username_pre = match_pre.group(1).strip(").,;:'\"")
            gh_repos_pre = fetch_github_projects(gh_username_pre)

    def normalize_pre(s):
        """Lowercase + strip everything except letters/numbers, for fuzzy comparison."""
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def extract_proj_names_pre(lines):
        """
        Turns raw "Projects" section lines into clean project *names* by
        filtering out bullet points/descriptions and stripping trailing
        junk like "| Tech Stack: ..." or "View Project" links.
        """
        names = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("-","•","*")): continue      # skip bullet points (these are usually descriptions, not titles)
            if len(stripped) < 5: continue                        # too short to be a real project name
            if stripped[0].islower(): continue                    # titles usually start with a capital letter
            desc_patterns = [r"^(and |the |to |for |with |by |in |of )",r"^(trends|insights|decisions|patterns|records)",r"^\d+\."]
            if any(re.match(p, stripped, re.I) for p in desc_patterns): continue
            clean = re.sub(r"\s*[|–-]\s*(Tech Stack|Tech|Stack|View Project).*","",stripped,flags=re.I)
            clean = re.sub(r"\s*View Project.*","",clean,flags=re.I)
            clean = re.sub(r"\s*Analyzed\s+[\d,+]+.*","",clean,flags=re.I)
            clean = re.sub(r"\s*Created\s+.*","",clean,flags=re.I)
            clean = clean.strip(" –-|:")
            if len(clean) > 4: names.append(clean)
        return names

    resume_proj_names_pre = extract_proj_names_pre(proj_raw_pre)
    resume_proj_count_pre = len(resume_proj_names_pre)
    github_proj_count_pre = len(gh_repos_pre)
    gh_norm   = [normalize_pre(r["name"]) for r in gh_repos_pre]
    res_norm  = [normalize_pre(n) for n in resume_proj_names_pre]
    # A "duplicate" = a resume project name that closely matches a GitHub repo
    # name (e.g. resume says "Weather App", GitHub repo is "weather-app").
    # We don't want to double-count those as 2 separate projects.
    dups_pre  = sum(1 for rn in res_norm if difflib.get_close_matches(rn, gh_norm, n=1, cutoff=0.7))
    total_proj_count = resume_proj_count_pre + github_proj_count_pre - dups_pre

    # ── Two-column layout: left = Skills/Education/Certs/Achievements,
    #                       right = Experience/Internship/Projects ──────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # SKILLS
        tags = data.get("_skill_tags", [])
        if tags:
            tags_html = "".join(f'<span class="skill-tag">{t}</span>' for t in tags)
            section_box("🛠️ Skills", tags_html)
        elif data.get("SKILLS"):
            section_box("🛠️ Skills", f'<p class="content-text">{chr(10).join(data["SKILLS"])}</p>')

        # EDUCATION
        edu = [l for l in data.get("EDUCATION", []) if l]
        if edu:
            entries = lines_to_entries(edu)
            html = ""
            for e in entries:
                html += '<div class="entry-block">'
                html += f'<div class="entry-title">{e[0]}</div>'
                for sub in e[1:]:
                    html += f'<div class="entry-body">{sub}</div>'
                html += "</div>"
            section_box("🎓 Education", html)

        # CERTIFICATIONS
        certs = [l for l in data.get("CERTIFICATIONS", []) if l]
        if certs:
            entries = lines_to_entries(certs)
            html = ""
            for e in entries:
                html += '<div class="entry-block">'
                html += f'<div class="entry-title">{e[0]}</div>'
                for sub in e[1:]:
                    html += f'<div class="entry-body">{sub}</div>'
                html += "</div>"
            section_box("🏅 Certifications", html)

        # ACHIEVEMENTS
        ach = [l for l in data.get("ACHIEVEMENTS", []) if l]
        if ach:
            html = "".join(f'<div class="entry-body">• {l}</div>' for l in ach)
            section_box("🏆 Achievements", html)

    with col2:
        # EXPERIENCE
        exp = [l for l in data.get("EXPERIENCE", []) if l]
        if exp:
            entries = lines_to_entries(exp)
            html = ""
            for e in entries:
                html += '<div class="entry-block">'
                html += f'<div class="entry-title">{e[0]}</div>'
                for sub in e[1:]:
                    html += f'<div class="entry-body">{sub}</div>'
                html += "</div>"
            section_box("💼 Work Experience", html)

        # INTERNSHIP
        intern = [l for l in data.get("INTERNSHIP", []) if l]
        if intern:
            entries = lines_to_entries(intern)
            html = ""
            for e in entries:
                html += '<div class="entry-block">'
                html += f'<div class="entry-title">{e[0]}</div>'
                for sub in e[1:]:
                    html += f'<div class="entry-body">{sub}</div>'
                html += "</div>"
            section_box("🏢 Internship", html)

        # ── PROJECTS (resume + GitHub) ─────────────────────────────────────
        # This mirrors the "_pre" block above, but this is the copy that
        # actually gets displayed to the user (with names/cards/badges).
        proj_raw = [l for l in data.get("PROJECTS", []) if l]

        def extract_project_names(lines):
            """Same cleanup logic as extract_proj_names_pre() above."""
            names = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("-") or stripped.startswith("•") or stripped.startswith("*"):
                    continue
                if len(stripped) < 5:
                    continue
                if stripped[0].islower():
                    continue
                desc_patterns = [
                    r"^(and |the |to |for |with |by |in |of )",
                    r"^(trends|insights|decisions|patterns|records)",
                    r"^\d+\.",
                ]
                if any(re.match(p, stripped, re.I) for p in desc_patterns):
                    continue
                clean = re.sub(r"\s*[|–-]\s*(Tech Stack|Tech|Stack|View Project).*", "", stripped, flags=re.I)
                clean = re.sub(r"\s*View Project.*", "", clean, flags=re.I)
                clean = re.sub(r"\s*Analyzed\s+[\d,+]+.*", "", clean, flags=re.I)
                clean = re.sub(r"\s*Created\s+.*", "", clean, flags=re.I)
                clean = clean.strip(" –-|:")
                if len(clean) > 4:
                    names.append(clean)
            return names

        resume_proj_names = extract_project_names(proj_raw)
        resume_proj_count = len(resume_proj_names)

        # Fetch GitHub repos (again — see note above about the "_pre" duplicate block)
        gh_repos    = []
        gh_links    = data["_contact"].get("GitHub", [])
        gh_username = None
        if gh_links:
            match = re.search(r"github\.com/([^/\s]+)", gh_links[0], re.I)
            if match:
                gh_username = match.group(1).strip(").,;:'\"")
                with st.spinner(f"🐙 Fetching GitHub repos for @{gh_username}…"):
                    gh_repos = fetch_github_projects(gh_username)

        github_proj_count = len(gh_repos)

        # ── Deduplicate: if resume project matches a GitHub repo, count as 1
        import difflib
        def normalize(s):
            return re.sub(r"[^a-z0-9]", "", s.lower())

        gh_normalized = [normalize(r["name"]) for r in gh_repos]
        resume_normalized = [normalize(n) for n in resume_proj_names]

        duplicates = 0
        for rn in resume_normalized:
            matches = difflib.get_close_matches(rn, gh_normalized, n=1, cutoff=0.7)
            if matches:
                duplicates += 1

        total_proj_count = resume_proj_count + github_proj_count - duplicates

        if proj_raw or gh_repos:
            # ── Stats bar: shows Resume Projects / GitHub Repos / Total ─────
            st.markdown(f"""
            <div class="section-wrap">
            <div class="sec-title">🚀 Projects</div>
            <div class="proj-stats-bar">
                <div class="proj-stat-box">
                    <div class="proj-stat-num">📄 {resume_proj_count}</div>
                    <div class="proj-stat-label">Resume Projects</div>
                </div>
                <div class="proj-stat-divider"></div>
                <div class="proj-stat-box">
                    <div class="proj-stat-num">🐙 {github_proj_count}</div>
                    <div class="proj-stat-label">GitHub Repos</div>
                </div>
                <div class="proj-stat-divider"></div>
                <div class="proj-stat-box">
                    <div class="proj-stat-num">✨ {total_proj_count}</div>
                    <div class="proj-stat-label">Total Unique Projects</div>
                </div>
            </div>
            <div class="gh-divider">📋 All Project Names</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Render each resume-listed project as its own row/card ───────
            for i, name_r in enumerate(resume_proj_names, 1):
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;
                            padding:0.55rem 0.8rem;margin-bottom:0.45rem;
                            background:#f1f5f9;border-radius:8px;
                            border-left:4px solid #6366f1;">
                    <span style="font-size:0.78rem;font-weight:700;color:#6366f1;min-width:24px;">#{i}</span>
                    <span style="font-size:0.88rem;color:#1e293b;font-weight:600;">📄 {name_r}</span>
                    <span style="margin-left:auto;font-size:0.72rem;background:#e0e7ff;
                                 color:#4338ca;border-radius:20px;padding:2px 10px;">Resume</span>
                </div>""", unsafe_allow_html=True)

            # ── Render each GitHub repo as its own row/card ──────────────────
            for j, repo in enumerate(gh_repos, resume_proj_count + 1):
                repo_url  = repo["url"]
                repo_name = repo["name"]
                repo_lang = repo.get("language") or ""
                # Build badges as a plain string — no nested HTML, no f-string conflicts
                badges = ""
                if repo_lang.strip():
                    badges += (
                        '<span style="font-size:0.72rem;background:#d1fae5;color:#065f46;'
                        f'border-radius:20px;padding:2px 8px;margin-right:4px;">{repo_lang}</span>'
                    )
                badges += (
                    '<span style="font-size:0.72rem;background:#d1fae5;color:#065f46;'
                    'border-radius:20px;padding:2px 10px;">GitHub</span>'
                )
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:0.6rem;'
                    'padding:0.55rem 0.8rem;margin-bottom:0.45rem;'
                    'background:#f8fafc;border-radius:8px;border-left:4px solid #10b981;">'
                    f'<span style="font-size:0.78rem;font-weight:700;color:#10b981;min-width:24px;">#{j}</span>'
                    f'<a href="{repo_url}" target="_blank" '
                    'style="font-size:0.88rem;color:#1e293b;font-weight:600;text-decoration:none;">'
                    f'🐙 {repo_name}</a>'
                    f'<span style="margin-left:auto;display:flex;gap:0.3rem;align-items:center;">{badges}</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

            # ── Explain to the user WHY GitHub repos might be missing ───────
            if not gh_repos and gh_username:
                # We found a username, but the API call returned nothing.
                st.markdown('<p style="color:#ef4444;font-size:0.85rem;">⚠️ Found GitHub link for @'
                             f'{gh_username} but could not fetch repos (private profile, no public repos, or API rate limit — GitHub allows 60 requests/hour without a token).</p>', unsafe_allow_html=True)
            elif not gh_repos and not gh_username and not gh_links:
                # We never even found a GitHub link/handle in the resume.
                st.markdown('<p style="color:#f59e0b;font-size:0.85rem;">ℹ️ No GitHub link was detected in this resume (no visible URL and no clickable hyperlink to github.com found).</p>', unsafe_allow_html=True)

    # ── Raw text (collapsible, for debugging / transparency) ──────────────────
    with st.expander("📃 View Raw Extracted Text"):
        st.text(raw_text)

    # ── JSON export: package everything up into a downloadable file ───────────
    export = {
        "name": data["_name"],
        "contact": data["_contact"],
        "summary": summary,
        "skills": data.get("_skill_tags", []),
        "education": edu if 'edu' in dir() else [],
        "experience": exp if 'exp' in dir() else [],
        "internship": intern if 'intern' in dir() else [],
        "projects": proj_raw if 'proj_raw' in dir() else [],
        "github_repos": [r["name"] for r in gh_repos] if gh_repos else [],
        "project_counts": {
            "resume": resume_proj_count,
            "github": github_proj_count,
            "total": total_proj_count,
        },
        "certifications": certs if 'certs' in dir() else [],
        "achievements": ach if 'ach' in dir() else [],
    }
    st.download_button(
        "⬇️ Download Parsed Data as JSON",
        data=json.dumps(export, indent=2),
        file_name="parsed_resume.json",
        mime="application/json",
    )

    # ========================================================================
    # STEP 3: MENTOR RATING PREDICTION (machine learning model)
    # ------------------------------------------------------------------------
    # We take 4 numeric features about this person and feed them into a
    # pre-trained XGBoost model (mentor_model.json) to predict a rating
    # out of 10:
    #   - experience_years : real years of work/internship experience,
    #                        computed from actual dates found on the resume
    #   - skills_count     : how many individual skills were listed
    #   - projects_count   : resume projects + GitHub repos (de-duplicated)
    #   - feedback_score   : defaults to 3.0 (neutral) since a brand-new
    #                        mentor has no student feedback yet
    # ========================================================================
    st.markdown("---")
    st.markdown("## Mentor Rating Prediction")

    import os
    import xgboost as xgb
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mentor_model.json")
    try:
        # Using the raw XGBoost Booster API (not the sklearn wrapper) so this
        # works even if scikit-learn isn't installed in this environment.
        ml_model = xgb.Booster()
        ml_model.load_model(model_path)

        skills_count_ml   = len(data.get("_skill_tags", []))
        projects_count_ml = total_proj_count

        # Look at both Experience and Internship text together when hunting
        # for date ranges (e.g. someone's only work history might be listed
        # as internships).
        exp_and_intern_text = "\n".join(
            data.get("EXPERIENCE", []) + data.get("INTERNSHIP", [])
        )
        detected_years, exp_detail = extract_experience_years(exp_and_intern_text)
        if detected_years is not None:
            experience_years = detected_years
        else:
            # Fallback if no parsable dates were found anywhere in the resume:
            # rough estimate from how much experience text exists, clearly
            # flagged as an estimate rather than presented as exact.
            exp_lines_ml = [l for l in data.get("EXPERIENCE", []) if l]
            experience_years = max(1, len(exp_lines_ml) // 3)
            exp_detail = "estimated — no explicit dates found on resume"
        feedback_score    = 3.0  # default neutral for new mentor

        # Build the exact same 4-column table the model was trained on
        input_data = pd.DataFrame([{
            "experience_years": experience_years,
            "skills_count"    : skills_count_ml,
            "projects_count"  : projects_count_ml,
            "feedback_score"  : feedback_score
        }])

        dmatrix = xgb.DMatrix(input_data)
        predicted_rating = ml_model.predict(dmatrix)[0]
        # Clamp to the valid 1-10 range in case the model extrapolates slightly outside it
        predicted_rating = max(1.0, min(float(predicted_rating), 10.0))

        # Turn the numeric score into a friendly grade label + color
        if predicted_rating >= 8:
            grade, color = "Excellent", "#10b981"
        elif predicted_rating >= 6:
            grade, color = "Good", "#6366f1"
        elif predicted_rating >= 4:
            grade, color = "Average", "#f59e0b"
        else:
            grade, color = "Poor", "#ef4444"

        st.markdown(f"""
        <div style="background:{color};border-radius:14px;padding:2rem;
                    text-align:center;margin-top:1rem;">
            <div style="font-size:3rem;font-weight:900;color:white;">{predicted_rating:.1f} / 10</div>
            <div style="font-size:1.4rem;color:white;font-weight:600;margin-top:0.5rem;">{grade} Mentor</div>
            <div style="font-size:0.85rem;color:rgba(255,255,255,0.8);margin-top:0.8rem;">
                Experience: {experience_years} yrs &nbsp;|&nbsp;
                Skills: {skills_count_ml} &nbsp;|&nbsp;
                Projects: {projects_count_ml}
            </div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.65);margin-top:0.3rem;">
                (Experience based on: {exp_detail})
            </div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.6);margin-top:0.4rem;">
                Based on resume data only (no student feedback yet)
            </div>
        </div>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(f"mentor_model.json not found at: {model_path}")
    except Exception as e:
        st.error(f"Could not load mentor_model.json: {e}")