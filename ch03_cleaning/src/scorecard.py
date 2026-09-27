"""Data-quality scorecard (Ch.3, code:scorecard). Reused in later chapters.

    from src.scorecard import Scorecard
    sc = Scorecard("galaxy10_cleaned.csv", reviewer="...", commit="abc1234")
    sc.score("completeness", "missing values: type and cause identified", 2, "MAR on r_mag")
    print(sc.report())
"""
from dataclasses import dataclass, field
from datetime import date

ITEMS = {
    "completeness": ["missing values: type and cause identified",
                     "sample selection bias assessed",
                     "coverage of the parameter space"],
    "accuracy": ["calibration source and date",
                 "systematic uncertainty estimated",
                 "at least one sanity check against known physics"],
    "consistency": ["units stated and consistent for every column",
                    "dtypes and naming consistent"],
    "timeliness": ["dataset version and last-update date"],
    "reproducibility": ["cleaning script and seed committed to git",
                        "requirements.txt and environment documented",
                        "notebook passes Restart & Run All"],
}


@dataclass
class Scorecard:
    dataset: str
    reviewer: str = "<name>"
    commit: str = "<hash>"
    scores: dict = field(default_factory=dict)          # (section, item) -> (score, evidence)

    def score(self, section, item, value, evidence=""):
        assert item in ITEMS[section], f"unknown item: {item}"
        assert value in (0, 1, 2, 3), "each item is scored 0-3"
        self.scores[(section, item)] = (value, evidence)

    def total(self):
        return sum(v for v, _ in self.scores.values())

    def report(self):
        lines = [f"DATA QUALITY SCORECARD - {self.dataset}",
                 f"Reviewer: {self.reviewer}     Date: {date.today()}     git commit: {self.commit}", ""]
        for i, (sec, items) in enumerate(ITEMS.items(), 1):
            got = sum(self.scores.get((sec, it), (0, ""))[0] for it in items)
            lines.append(f"[{i}] {sec.upper():58s}{got:2d}/{3 * len(items)}")
            for it in items:
                v, ev = self.scores.get((sec, it), (None, ""))
                box = f"[{v}]" if v is not None else "[ ]"
                lines.append(f"  {box} {it:54s}(0-3)" + (f"  {ev}" if ev else ""))
        maxi = 3 * sum(len(v) for v in ITEMS.values())
        lines += ["", f"TOTAL SCORE: {self.total():44d} / {maxi}"]
        return "\n".join(lines)
