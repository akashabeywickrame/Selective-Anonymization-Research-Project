# =============================================================================
# utils/export_thesis_chapters.py — Convert chapters_4_5_6.tex to the
# University of Peradeniya thesis template (uopThesis class)
# =============================================================================
#
# The working file chapters_4_5_6.tex is written article-style (\section for a
# chapter, hard-coded "Section~4.1" numbers, \cite). The university template is
# a book-style class with one file per chapter in chapters/, natbib author-year
# citations and a BibTeX References.bib. This script writes:
#
#   thesis/chapters/Results.tex
#   thesis/chapters/Discussion.tex
#   thesis/chapters/Conclusion.tex
#   thesis/References_chapters_4_6.bib   (entries cited by these chapters)
#
# Re-run it after every edit to chapters_4_5_6.tex:
#   python -m utils.export_thesis_chapters
# =============================================================================

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "chapters_4_5_6.tex"
OUT = ROOT / "thesis" / "chapters"

# Hard-coded section numbers -> labels (so numbering follows the template)
SECTION_REFS = {
    "Section~4.1": r"Section~\ref{sec:dataset}",
    "Section~4.2": r"Section~\ref{sec:results_2a}",
    "Section~4.3": r"Section~\ref{sec:results_2b}",
    "Section~4.4": r"Section~\ref{sec:results_clips}",
    "Section~4.5": r"Section~\ref{sec:results_secondpass}",
    "Section~4.6": r"Section~\ref{sec:baseline_comparison}",
    "Section~4.7": r"Section~\ref{sec:fps}",
    "Section~5.3": r"Section~\ref{sec:disc_fps}",
}

BIB = r"""% BibTeX entries cited in the Results, Discussion and Conclusion chapters.
% Append to References.bib (skip any key that is already there).

@inproceedings{bystandar,
  author    = {Corbett, Matthew and David-John, Brendan and Shang, Jiacheng and Hu, Y. Charlie and Ji, Bo},
  title     = {{BystandAR}: Protecting Bystander Visual Data in Augmented Reality Systems},
  booktitle = {Proceedings of the 21st Annual International Conference on Mobile Systems, Applications and Services (MobiSys '23)},
  publisher = {ACM},
  pages     = {370--382},
  year      = {2023},
  doi       = {10.1145/3581791.3596830}
}

@article{egoblur2023,
  author  = {{Meta Reality Labs}},
  title   = {{EgoBlur}: Privacy-Preserving Head-Mounted Egocentric Vision},
  journal = {arXiv preprint arXiv:2308.13093},
  year    = {2023}
}

@inproceedings{ref:safear2025,
  author    = {Costa, R. L. de C. and Marto, A. and Santos, L. and Rabad{\~a}o, C.},
  title     = {{SafeAR}: Privacy-Maintenance in Augmented Reality Applications},
  booktitle = {Proceedings of the 11th European Conference on Service-Oriented and Cloud Computing (ESOCC 2025)},
  series    = {Lecture Notes in Computer Science},
  volume    = {15547},
  pages     = {146--151},
  year      = {2025}
}

@inproceedings{ref:bystandaria,
  author    = {Jad Al Aaraj, A. M.},
  title     = {{BystandARIA}: Enabling {AR} Bystander Privacy Using {LEDs}},
  booktitle = {Proceedings of the Twenty-sixth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc '25)},
  address   = {Houston, TX, USA},
  pages     = {444--449},
  year      = {2025},
  doi       = {10.1145/3704413.3765296}
}

@misc{ref:qcom_ar1,
  author       = {{Qualcomm Technologies}},
  title        = {Snapdragon {AR1} Gen 1 Platform},
  howpublished = {\url{https://www.qualcomm.com/products/mobile/snapdragon/xr-vr-ar/snapdragon-ar1-gen-1-platform}},
  year         = {2023},
  note         = {Product page, accessed October 2026}
}

@misc{ref:gsmarena_display,
  author       = {{GSMArena}},
  title        = {Meta Unveils {Ray-Ban Display} and {Ray-Ban Meta (Gen 2)} Smart Glasses},
  howpublished = {\url{https://www.gsmarena.com/meta_rayban_display_and_rayban_meta_gen_2_smart_glasses_debut-news-69560.php}},
  year         = {2025},
  note         = {18 September 2025, accessed October 2026}
}

@misc{ref:spectacles5,
  author       = {{Snap Inc.}},
  title        = {{SPS} 2024: Introducing the New {Spectacles} and {Snap OS}},
  howpublished = {\url{https://newsroom.snap.com/sps-2024-spectacles-snapos}},
  year         = {2024},
  note         = {Snap Newsroom, 17 September 2024, accessed October 2026}
}

@misc{ref:aria2,
  author       = {{Meta}},
  title        = {Introducing {Aria Gen 2}: Unlocking New Research in Machine Perception, Contextual {AI}, Robotics, and More},
  howpublished = {\url{https://www.meta.com/blog/project-aria-gen-2-next-generation-egocentric-research-glasses-reality-labs-ai-robotics/}},
  year         = {2025},
  note         = {Accessed October 2026}
}

@misc{ref:ultralytics_qnn,
  author       = {{Ultralytics}},
  title        = {Qualcomm {QNN} Export for {Ultralytics YOLO} Models},
  howpublished = {\url{https://docs.ultralytics.com/integrations/qnn}},
  year         = {2026},
  note         = {Documentation, accessed October 2026}
}

@misc{ref:qcom_facedet,
  author       = {{Qualcomm AI Hub}},
  title        = {Lightweight-Face-Detection-Quantized},
  howpublished = {\url{https://huggingface.co/qualcomm/Lightweight-Face-Detection-Quantized}},
  year         = {2026},
  note         = {Model card, Hugging Face, accessed October 2026}
}

@misc{ref:meta_bystander,
  author       = {{Meta}},
  title        = {Bystander Privacy},
  howpublished = {\url{https://about.fb.com/wp-content/uploads/2026/07/Bystander-Privacy.pdf}},
  year         = {2026},
  note         = {White paper, accessed October 2026}
}
"""


def convert(text):
    # Heading levels: article-style chapter -> book-style chapter
    text = re.sub(r"^\\subsection\{", r"\\section{", text, flags=re.M)
    text = re.sub(r"^\\section\{(Results|Discussion|Conclusion)\}", r"\\chapter{\1}", text, flags=re.M)
    for old, new in SECTION_REFS.items():
        text = text.replace(old, new)
    # natbib author-year: parenthetical citations
    text = text.replace(r"\cite{", r"\citep{")
    return text.strip() + "\n"


def main():
    s = SRC.read_text(encoding="utf-8")
    s = s.split("%  NEW REFERENCES")[0]                 # bibliography goes to the .bib
    s = s.rsplit("% =====", 1)[0] if s.rstrip().endswith("=") else s
    i_res = s.index(r"\section{Results}")
    i_dis = s.index(r"\section{Discussion}")
    i_con = s.index(r"\section{Conclusion}")
    helper = s[:i_res]
    helper = helper[helper.index(r"% Shrinks"):] if r"% Shrinks" in helper else ""

    def strip_banner(t):
        # drop the trailing "% ===== CHAPTER N =====" banner that precedes the next chapter
        return re.sub(r"(\n%\s*=+\n%\s*CHAPTER.*\n%\s*=+\s*)$", "\n", t.rstrip() + "\n")

    parts = {
        "Results": helper + strip_banner(s[i_res:i_dis]),
        "Discussion": strip_banner(s[i_dis:i_con]),
        "Conclusion": strip_banner(s[i_con:]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in parts.items():
        (OUT / f"{name}.tex").write_text(convert(body), encoding="utf-8")
        print(f"[Export] thesis/chapters/{name}.tex")
    (ROOT / "thesis" / "References_chapters_4_6.bib").write_text(BIB, encoding="utf-8")
    print("[Export] thesis/References_chapters_4_6.bib")


if __name__ == "__main__":
    main()
