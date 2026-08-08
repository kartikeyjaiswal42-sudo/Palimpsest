"""The corpus source registry: where every essay came from, and on what terms.

Redistribution policy
---------------------
This repository does **not** commit the human essay text. Most of it is other people's
writing under licences that do not permit redistribution -- published admissions essays are
copyright the colleges that published them, and PERSUADE/ELLIPSE are CC BY-NC-SA. What is
committed is this registry plus ``scripts/fetch_corpus.py``, so anyone can rebuild the
identical corpus, and a manifest of SHA-256 hashes so they can prove they rebuilt the
*identical* one.

The machine half is different: we generated it, so it ships in the repository.

Every source below carries ``verified``, recording that the URL was actually fetched and
its shape confirmed rather than assumed. Sources we investigated and could not use are in
``docs/02-dataset.md`` with the reason, because a dead end other people will also hit is
worth writing down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Source", "SOURCES", "SOURCES_BY_ID", "LIANG_BASE", "GHOSTBUSTER_BASE"]

LIANG_BASE = (
    "https://raw.githubusercontent.com/Weixin-Liang/ChatGPT-Detector-Bias/"
    "main/Data_and_Results"
)
GHOSTBUSTER_BASE = (
    "https://raw.githubusercontent.com/vivek3141/ghostbuster-data/master/essay/human"
)


@dataclass(frozen=True)
class Source:
    """One corpus source, with the provenance we can defend in writing."""

    id: str
    label: str
    #: "human" | "machine" | "hybrid"
    authorship: str
    #: What role it plays: "train", "esl-eval", "unseen-generator", "reference", "ablation".
    role: str
    url: str
    fetcher: str
    licence: str
    citation: str
    #: Honest statement of whether the text provably predates ChatGPT (2022-11-30).
    pre_chatgpt: str
    #: What this source does NOT cover. Read this before trusting a number computed on it.
    limitations: str
    expected_n: int
    redistributable: bool
    verified: bool = True
    extra: dict = field(default_factory=dict)


SOURCES: tuple[Source, ...] = (
    # ---------------------------------------------------------------- human, in-domain
    Source(
        id="liang_college_human",
        label="Liang et al. — real US college admissions essays",
        authorship="human",
        role="train",
        url=f"{LIANG_BASE}/Human_Data/CollegeEssay_real_70/data.json",
        fetcher="liang_json",
        licence="No LICENSE file in repo (README shows an MIT badge; GitHub API reports "
        "license:null). Underlying essays are reproduced from college admissions sites and "
        "are copyright their authors/institutions. Research use only; not redistributed here.",
        citation="Liang, Yuksekgonul, Mao, Wu & Zou (2023). GPT detectors are biased "
        "against non-native English writers. Patterns 4(7):100779.",
        pre_chatgpt="Collected before 2023-04-05 (repo creation). No per-essay dates exist, "
        "so this is 'almost certainly pre-ChatGPT' rather than provable per document.",
        limitations="Sourced from essays colleges chose to publish, so it is a sample of "
        "*successful, edited* essays -- not representative of a typical applicant's draft. "
        "Skews US, native-English, and selective institutions.",
        expected_n=70,
        redistributable=False,
    ),
    Source(
        id="liang_hewlett_human",
        label="Liang et al. — Hewlett ASAP 8th-grade essays (2012)",
        authorship="human",
        role="train",
        url=f"{LIANG_BASE}/Human_Data/HewlettStudentEssay_real_88/data.json",
        fetcher="liang_json",
        licence="Upstream Hewlett/ASAP Kaggle competition terms. Not redistributed here.",
        citation="Hewlett Foundation Automated Student Assessment Prize (2012), via "
        "Liang et al. (2023).",
        pre_chatgpt="YES, provably. Written for the 2012 ASAP competition, a decade before "
        "ChatGPT. This is the strongest human-authorship guarantee in the corpus.",
        limitations="8th-grade classroom writing, not admissions essays. Register and "
        "maturity differ; it anchors 'certainly human' but is off-domain for the product.",
        expected_n=88,
        redistributable=False,
    ),
    Source(
        id="hamilton",
        label="Hamilton College — Essays That Worked (2003, 2008, 2014, 2022)",
        authorship="human",
        role="train",
        url="https://www.hamilton.edu/admission/apply/college-essays-that-worked/",
        fetcher="hamilton_html",
        licence="Copyright Hamilton College; the 2003 page states essays are 'reprinted "
        "with their permission'. Research use only; not redistributed here.",
        citation="Hamilton College Admission, Essays That Worked archives.",
        pre_chatgpt="YES for all four archives. The URL year is the publication year and "
        "each page names its cohort (2022 archive = Class of 2026, written in the "
        "autumn-2021 cycle, roughly a year before ChatGPT).",
        limitations="30 essays only. Published-as-exemplary, so the same selection bias as "
        "the Liang college set.",
        expected_n=30,
        redistributable=False,
        extra={"years": ["2003", "2008", "2014", "2022"]},
    ),
    Source(
        id="jhu",
        label="Johns Hopkins — Essays That Worked (pre-2022-11-30 subset)",
        authorship="human",
        role="train",
        url="https://apply.jhu.edu/wp-json/wp/v2/insider_article",
        fetcher="jhu_wp_api",
        licence="Copyright Johns Hopkins University. Research use only; not redistributed.",
        citation="Johns Hopkins Undergraduate Admissions, Essays That Worked.",
        pre_chatgpt="Filtered to publication date <= 2022-11-30, which yields 31 of 59. "
        "TRAP: the year TAGS are graduating-class labels roughly four years ahead of "
        "publication (tag 'essays-that-worked-2020' returns posts dated 2016-12-15). We "
        "filter on the API `date` field, never on the tag.",
        limitations="Same publication-selection bias. Attributed by first name and class "
        "year only.",
        expected_n=31,
        redistributable=False,
    ),
    # ---------------------------------------------------------------- human, ESL evaluation
    Source(
        id="liang_toefl",
        label="Liang et al. — TOEFL essays by non-native writers",
        authorship="human",
        role="esl-eval",
        url=f"{LIANG_BASE}/Human_Data/TOEFL_real_91/data.json",
        fetcher="liang_json",
        licence="Scraped by the original authors from a Chinese educational forum; no "
        "upstream licence stated. Research use only; not redistributed here.",
        citation="Liang et al. (2023), Patterns 4(7):100779.",
        pre_chatgpt="Collected before 2023-04-05. No per-essay dates.",
        limitations="CRITICAL LENGTH CONFOUND: median 104 words against 633 for the college "
        "essays. Any ESL comparison that does not length-match is measuring length, not "
        "language background. We length-match explicitly -- see docs/05-esl.md.",
        expected_n=91,
        redistributable=False,
    ),
    Source(
        id="ellipse",
        label="ELLIPSE — English Language Learner essays with graded proficiency",
        authorship="human",
        role="esl-eval",
        url="https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/"
        "ELLIPSE_Final_github_train.csv",
        fetcher="ellipse_csv",
        licence="CC BY-NC-SA 4.0. Attribution required, non-commercial, share-alike. "
        "Not redistributed here.",
        citation="Crossley, Tian, Baffour, Franklin, Kim, Morris, Benner, Picou & Boser "
        "(2023). International Journal of Learner Corpus Research 9(2), 248-269.",
        pre_chatgpt="YES. Same 2010-2020 US school-assessment pool as PERSUADE; the Kaggle "
        "competition using it closed in November 2022, before ChatGPT launched.",
        limitations="100% ELL writers, so it has no internal native-speaker control -- it "
        "must be paired with a native set. Argumentative school essays, not admissions "
        "essays, so register differs from the product's target domain.",
        expected_n=3911,
        redistributable=False,
        extra={"text_column": "full_text", "proficiency_column": "Overall"},
    ),
    Source(
        id="persuade",
        label="PERSUADE 2.0 — student argumentative essays with an ELL flag",
        authorship="human",
        role="esl-eval",
        url="https://datasets-server.huggingface.co/rows?dataset=nlpatunt%2FD_persuade_2"
        "&config=default&split=train",
        fetcher="persuade_hf_rows",
        licence="CC BY-NC-SA 4.0. Non-commercial and share-alike. Not redistributed here.",
        citation="Crossley, Baffour, Tian, Franklin, Benner & Boser (2024). A large-scale "
        "corpus for assessing written argumentation: PERSUADE 2.0. Assessing Writing 61.",
        pre_chatgpt="YES. Essays written 2010-2020 by US students in grades 6-12.",
        limitations="The single best ESL control we have -- same prompts, same graders, "
        "same cohort, only the ELL flag differs -- but the domain is argumentative school "
        "writing, not admissions essays. ~9% of rows are ELL.",
        expected_n=2000,
        redistributable=False,
        extra={"ell_column": "ell_status", "text_column": "full_text"},
    ),
    # ---------------------------------------------------------------- machine, unseen family
    Source(
        id="liang_college_gpt3",
        label="Liang et al. — GPT-3.5 admissions essays (held out as an unseen generator)",
        authorship="machine",
        role="unseen-generator",
        url=f"{LIANG_BASE}/GPT_Data/CollegeEssay_gpt3_31/data.json",
        fetcher="liang_json",
        licence="Generated by the original authors with GPT-3.5. Not redistributed here.",
        citation="Liang et al. (2023), Patterns 4(7):100779.",
        pre_chatgpt="N/A -- machine text, generated in early 2023.",
        limitations="Median 261 words against 633 for the human college essays. Training on "
        "this pair directly would teach the detector that short means machine. It is "
        "therefore NEVER trained on -- only used as a held-out test of whether a detector "
        "fitted on our own generations transfers to a different model family.",
        expected_n=31,
        redistributable=False,
    ),
    Source(
        id="liang_college_gpt3_prompteng",
        label="Liang et al. — GPT-3.5 admissions essays with prompt engineering",
        authorship="machine",
        role="unseen-generator",
        url=f"{LIANG_BASE}/GPT_Data/CollegeEssay_gpt3PromptEng_31/data.json",
        fetcher="liang_json",
        licence="Generated by the original authors. Not redistributed here.",
        citation="Liang et al. (2023), Patterns 4(7):100779.",
        pre_chatgpt="N/A -- machine text.",
        limitations="Same length caveat as above. Prompt-engineered to evade detection, so "
        "it is the harder of the two unseen-generator sets and we report it separately.",
        expected_n=31,
        redistributable=False,
    ),
    # ---------------------------------------------------------------- ablation pairs
    Source(
        id="liang_toefl_gpt4polished",
        label="Liang et al. — the same 91 TOEFL essays, GPT-4 polished",
        authorship="hybrid",
        role="ablation",
        url=f"{LIANG_BASE}/Human_Data/TOEFL_gpt4polished_91/data.json",
        fetcher="liang_json",
        licence="Derived by the original authors. Not redistributed here.",
        citation="Liang et al. (2023), Patterns 4(7):100779.",
        pre_chatgpt="N/A -- human content, machine surface.",
        limitations="A controlled probe rather than a natural sample: identical content, "
        "identical author, only the surface changed. That is exactly what makes it useful "
        "for isolating what our features respond to.",
        expected_n=91,
        redistributable=False,
    ),
    Source(
        id="liang_hewlett_gptsimplify",
        label="Liang et al. — the same 88 ASAP essays, GPT-simplified to read non-native",
        authorship="hybrid",
        role="ablation",
        url=f"{LIANG_BASE}/Human_Data/HewlettStudentEssay_GPTsimplify_88/data.json",
        fetcher="liang_json",
        licence="Derived by the original authors. Not redistributed here.",
        citation="Liang et al. (2023), Patterns 4(7):100779.",
        pre_chatgpt="N/A.",
        limitations="The mirror image of the polish ablation: same content, surface made "
        "*less* fluent. In the source paper this moved commercial detectors' false-positive "
        "rate from 5.2% to 56.7%, which is the cleanest causal evidence available that "
        "these detectors key on fluency rather than authorship.",
        expected_n=88,
        redistributable=False,
    ),
    # ---------------------------------------------------------------- n-gram reference
    Source(
        id="ghostbuster_ivypanda",
        label="Ghostbuster — 1,000 IvyPanda student essays (n-gram reference only)",
        authorship="human",
        role="reference",
        url=GHOSTBUSTER_BASE,
        fetcher="ghostbuster_txt",
        licence="CC BY 3.0 Unported -- the cleanest licence in the corpus. Attribution "
        "required, commercial use permitted.",
        citation="Verma, Fleisig, Tomlin & Klein (2024). Ghostbuster: Detecting Text "
        "Ghostwritten by Large Language Models. NAACL 2024. arXiv:2305.15047.",
        pre_chatgpt="PARTIAL, and this is the honest weak point. The upstream HuggingFace "
        "dataset was assembled 2023-01-23, eight weeks after ChatGPT launched, from "
        "IvyPanda's back-catalogue. Individual essays are near-certainly older but carry no "
        "dates. It is therefore used ONLY to fit the n-gram reference and never as a "
        "labelled human example -- a little contamination in a background frequency model "
        "is tolerable in a way that contaminated training labels are not.",
        limitations="Essay-mill content, so it is 'student-style writing' rather than "
        "genuine student writing. Predominantly British and American English.",
        expected_n=1000,
        redistributable=True,
    ),
)

SOURCES_BY_ID: dict[str, Source] = {s.id: s for s in SOURCES}
