# DeterministicQueryEngine.py — v1.4.6 (period-aware, subgroup-first, orientation-aware)
# VERSION_NOTE:
# - Fixes stray/partial-patch regressions (no \1 artifacts; all if: have bodies)
# - Robust group routing with Sexual orientation/identity precedence over Sex
# - Dataset-aware orientation fallback → TOTAL with exact warning string
# - Subgroup-first ordering using query tokens (Bisexual, Female, Midwest, <100% FPL, Black, etc.)
# - Preserves existing children/AGE101 note behavior and period logic

import os
import re
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

ENGINE_VERSION = "1.4.6"

# --- BEGIN IMPORT BANNER ---
try:
    import sys as _sys, os as _os, time as _time, hashlib as _hashlib
    _fp = __file__
    try:
        _raw = open(_fp, 'rb').read()
        _md5 = _hashlib.md5(_raw).hexdigest()
    except Exception:
        _md5 = 'unknown'
    _mt = _time.ctime(_os.path.getmtime(_fp)) if _os.path.exists(_fp) else 'unknown'
    print(f"[ENGINE-IMPORT] version={ENGINE_VERSION} file={_fp} md5={_md5} mtime={_mt}", file=_sys.stderr)
except Exception as _e:
    try:
        print(f"[ENGINE-IMPORT] banner-failed: {_e.__class__.__name__}: {_e}", file=_sys.stderr)
    except Exception:
        pass
# --- END IMPORT BANNER ---

# ---------- Helpers ----------
def detect_mojibake(value) -> bool:
    try:
        s = str(value)
    except Exception:
        return False
    return ('\ufffd' in s) or ('�' in s) or ('\u00c2' in s) or ('Â' in s)

def scan_dataframe_mojibake(df: pd.DataFrame) -> Dict[str, Any]:
    result = {"total": 0, "by_column": {}}
    try:
        obj_cols = [c for c in getattr(df, "columns", []) if str(df[c].dtype) == "object"]
        for c in obj_cols:
            bad = int(df[c].astype(str).apply(detect_mojibake).sum())
            if bad:
                result["by_column"][c] = bad
                result["total"] += bad
    except Exception:
        pass
    return result

# ---------- Result type ----------
@dataclass
class QueryResult:
    query: str
    topic: Optional[str]
    dataset_id: Optional[str]
    group: Optional[str]
    subgroups: List[str]
    years: List[int]             # kept for back-compat (may be empty with period logic)
    data: List[Dict[str, Any]]
    summary: str
    confidence: float
    warnings: List[str]


class DeterministicQueryEngine:
    # === Mapping-driven demographic inference (lazy) ===
    def _ensure_demo_index(self, topic_df=None):
        """Build a small index from keyword_mappings2.json for group/subgroup hints."""
        if getattr(self, "_demo_index", None) is not None:
            return
        self._demo_index = {}
        self._group_priority = ["Sexual orientation", "Sexual identity", "Sex"]
        for cand in [os.getenv("KEYWORD_MAPPINGS_PATH",""), "keyword_mappings2.json", "/app/keyword_mappings2.json"]:
            if cand and os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        km = json.load(f)
                    dem = km.get("demographic_mappings", {})
                    for _major, groups in (dem or {}).items():
                        for _g, payload in (groups or {}).items():
                            cg = (payload.get("cdc_groups") or [None])[0]
                            if not cg:
                                continue
                            entry = self._demo_index.setdefault(cg, {"keywords": set(), "subgroups": set()})
                            for kw in payload.get("user_keywords", []) or []:
                                if kw:
                                    entry["keywords"].add(str(kw).lower())
                            for sg in payload.get("cdc_subgroups", []) or []:
                                if sg:
                                    entry["subgroups"].add(str(sg))
                    break
                except Exception:
                    pass

    def _infer_group_and_sub_from_mappings(self, clean_query: str, topic_df):
        self._ensure_demo_index(topic_df)
        q = (clean_query or "").lower()
        hits, req_sub = [], None
        if not self._demo_index:
            return hits, req_sub
        available = set(str(x) for x in topic_df["group"].unique())
        for gname, data in self._demo_index.items():
            if gname not in available:
                continue
            if any(kw in q for kw in data["keywords"]):
                hits.append(gname)
            for sg in data["subgroups"]:
                sgl = str(sg).lower()
                if sgl and sgl in q:
                    if gname not in hits:
                        hits.append(gname)
                    if req_sub is None:
                        req_sub = sg
        for pg in reversed(self._group_priority):
            if pg in hits:
                hits = [pg] + [h for h in hits if h != pg]
        return hits, req_sub

    # ---------- Init / Config ----------
    def __init__(self, data_path: str = "standardized_dqs_full.json"):
        # Strictness flags
        self.strict_topic_only = os.getenv("STRICT_TOPIC_ONLY", "false").lower() in ("1","true","yes","on")
        self.strict_no_implicit_group = os.getenv("STRICT_NO_IMPLICIT_GROUP", "false").lower() in ("1","true","yes","on")

        # Data & config
        self.master_df: Optional[pd.DataFrame] = None
        self.keyword_mappings: Dict[str, Any] = {}
        self.disambiguation_rules: Dict[str, Any] = {}
        self.group_label_map: Dict[str, Any] = {}
        self.dataset_map: Dict[str, Any] = {}
        self.topic_links: Dict[str, Any] = {}
        self.dataset_overrides: Dict[str, Any] = {}
        self.topic_index: Dict[str, set] = {}

        # Search dirs for config
        self._config_dirs: List[Path] = self._build_config_dirs(data_path)

        # Load
        self._load_config()
        if data_path and Path(data_path).exists():
            self._load_master_data(data_path)
            self._build_topic_index()

    # ---------- Config discovery ----------
    def _build_config_dirs(self, data_path: str) -> List[Path]:
        dirs: List[Path] = []
        try:
            dirs.append(Path.cwd())
        except Exception:
            pass
        for envk in ("DQS_CONFIG", "DATA_PATH", "DQS_MASTER"):
            p = os.getenv(envk, "").strip()
            if p:
                pp = Path(p)
                dirs.append(pp.parent if pp.is_file() else pp)
        for d in ("/config", "/srv/app", "/work", "/app"):
            dirs.append(Path(d))
        if data_path:
            dp = Path(data_path)
            if dp.exists():
                dirs.append(dp.parent)
        uniq: List[Path] = []
        seen = set()
        for d in dirs:
            try:
                rp = d.resolve()
            except Exception:
                continue
            if rp not in seen and rp.exists():
                seen.add(rp)
                uniq.append(rp)
        return uniq

    def _try_load_json(self, name: str, default: Any) -> Any:
        for d in self._config_dirs:
            fp = d / name
            try:
                if fp.exists():
                    with open(fp, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception:
                pass
        try:
            with open(name, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _load_config(self):
        self.disambiguation_rules = self._try_load_json("disambiguation_rules.json", {})
        self.group_label_map      = self._try_load_json("group_label_map.json", {})
        self.dataset_map          = self._try_load_json("dataset_map.json", {})
        self.topic_links          = self._try_load_json("cdc_topic_links.json", {})
        self.dataset_overrides    = self._try_load_json("dataset_overrides.json", {})
        km2 = self._try_load_json("keyword_mappings2.json", {})
        if not km2:
            km2 = {"topic_mappings": {}, "demographic_mappings": {}, "composite_group_mappings": {}}
        self.keyword_mappings = km2

    # ---------- Data load (adds period fields) ----------
    def _load_master_data(self, data_path: str):
        try:
            if data_path.endswith(".json"):
                df = pd.read_json(data_path)
            elif data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
            else:
                self.master_df = None
                return

            # Coerce numerics
            for col in ("estimate","estimate_lci","estimate_uci","standard_error"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Clean strings
            def _clean(s):
                if pd.isna(s): return ""
                return re.sub(r"\s+"," ",str(s)).strip()
            for col in ("group","subgroup","classification"):
                if col in df.columns:
                    df[col] = df[col].apply(_clean)
            if "topic" in df.columns:
                df["topic"] = df["topic"].apply(lambda x: str(x).strip() if pd.notna(x) else "")

            # Legacy numeric year (kept for back-compat)
            def _extract_year(tp):
                if pd.isna(tp): return None
                s = str(tp).strip()
                m = re.search(r"\b(19[89]\d|20\d{2})\b", s)
                return int(m.group(1)) if m else None
            df["time_period_original"] = df.get("time_period", "")
            df["time_period_numeric"]  = df.get("time_period","").apply(_extract_year)

            # Period parsing
            df["period_label"] = df.get("time_period", "").astype(str)
            def _parse_period_label(s: str):
                ys = [int(y) for y in re.findall(r'(19[89]\d|20\d{2})', str(s))]
                if not ys: return None, None
                ys.sort(); return ys[0], ys[-1]
            df[["period_start","period_end"]] = df["period_label"].apply(lambda s: pd.Series(_parse_period_label(s)))

            self.master_df = df
        except Exception as e:
            print("Error loading master data:", e)
            self.master_df = None

    def _build_topic_index(self):
        self.topic_index = {}
        if self.master_df is not None and "topic" in self.master_df.columns:
            for t in self.master_df["topic"].dropna().unique():
                nt = self.normalize_query(t)
                if nt:
                    self.topic_index.setdefault(nt, set()).add(t)

    # ---------- Text utils ----------
    def clean_text(self, t:str)->str:
        if not t: return ""
        t = str(t).replace('\ufffd','').replace('\u00c2','')
        try:
            t = unicodedata.normalize("NFKC", t)
        except Exception:
            pass
        return re.sub(r"\s+"," ",t).strip()

    def normalize_query(self, q:str)->str:
        q = self.clean_text(q).lower()
        stop = {'the','of','in','for','by','with','among','during'}
        return ' '.join(w for w in q.split() if w not in stop)

    _CANON_MAP = [
        (re.compile(r'^sex( of (adult|child))?$', re.I), 'Sex'),
        (re.compile(r'^race.*hispanic.*$', re.I), 'Race & Hispanic origin'),
        (re.compile(r'^race$', re.I), 'Race'),
        (re.compile(r'^age( group)?$', re.I), 'Age group'),
        (re.compile(r'^health insurance coverage$', re.I), 'Health insurance coverage'),
        (re.compile(r'^sexual orientation$', re.I), 'Sexual orientation'),
    ]
    def canonicalize_group(self, g: str) -> str:
        gs = (g or "").strip()
        for rx, canon in self._CANON_MAP:
            if rx.match(gs):
                return canon
        return gs

    # ---------- Topic matching ----------
    def match_topic_adult_first(self, query: str) -> Tuple[Optional[str], Optional[str], List[str]]:
        warnings: List[str] = []
        qn = self.normalize_query(query)
        wants_child = any(w in qn for w in ['child','children','kid','kids','pediatric'])
        wants_adult = any(w in qn for w in ['adult','adults','18']) and not wants_child

        # Exact
        exact = []
        for topic in self.topic_links.keys():
            if topic.lower().strip() in query.lower():
                exact.append(topic)
        if exact:
            chosen = None
            if wants_child:
                child = [t for t in exact if 'child' in t.lower()]
                chosen = child[0] if child else exact[0]
            elif wants_adult:
                adult = [t for t in exact if 'child' not in t.lower()]
                chosen = adult[0] if adult else exact[0]
            else:
                chosen = exact[0]
            did = None
            if chosen in self.topic_links:
                url = (self.topic_links[chosen] or [''])[0]
                if '/d/' in url:
                    did = url.split('/d/')[-1]
            return chosen, did, warnings

        # Substring / keyword maps
        candidates: List[str] = []
        for topic in self.topic_links.keys():
            if self.normalize_query(topic) in qn or qn in self.normalize_query(topic):
                candidates.append(topic)

        tm = self.keyword_mappings.get("topic_mappings",{})
        for key, val in tm.items():
            if isinstance(val, dict):
                for alias in val.get("user_keywords",[]):
                    if self.normalize_query(alias) in qn or qn in self.normalize_query(alias):
                        candidates.extend(val.get("cdc_topics",[]))
            else:
                try:
                    for alias in (val if isinstance(val, list) else []):
                        if self.normalize_query(alias) in qn or qn in self.normalize_query(alias):
                            candidates.append(key)
                except Exception:
                    pass

        # Fallback: scan topic names from data
        if not candidates and not self.strict_topic_only:
            for nt, variants in self.topic_index.items():
                if nt in qn or qn in nt:
                    candidates.extend(list(variants))

        if not candidates:
            return None, None, ["TP001: No topic matches found"]

        # Dedup preserve order
        seen=set(); uniq=[]
        for t in candidates:
            if t not in seen:
                seen.add(t); uniq.append(t)
        candidates = uniq

        # Re-rank candidates by token overlap to avoid cross-topic mismatches
        def _toks(s):
            return set(re.findall(r"[a-z0-9]+", self.normalize_query(s)))
        qt = _toks(query)
        qt -= {'child','children','kid','kids','adult','adults','men','women','male','female'}
        scored = []
        for t in candidates:
            tt = _toks(t)
            score = len(qt & tt)
            if wants_child and 'child' in t.lower():
                score += 0.5
            if wants_adult and 'child' not in t.lower():
                score += 0.25
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [t for s,t in scored if s > 0] or [t for s,t in scored]

        if wants_child:
            child = [t for t in candidates if 'child' in t.lower()]
            topic = child[0] if child else candidates[0]
        elif wants_adult:
            adult = [t for t in candidates if 'child' not in t.lower()]
            topic = adult[0] if adult else candidates[0]
        else:
            topic = candidates[0]

        did = None
        if topic in self.topic_links:
            url = (self.topic_links[topic] or [''])[0]
            if '/d/' in url:
                did = url.split('/d/')[-1]

        # Child note if user asked for kids but adult-only selected
        try:
            _is_child_topic = 'child' in (topic or '').lower()
        except Exception:
            _is_child_topic = False
        _child_dsids = {'7ctq-myvs'}  # NHIS Children Summary Statistics
        if wants_child and (not _is_child_topic) and (did is None or did not in _child_dsids):
            warnings.append("AGE101: child-estimates-unavailable-using-adult")

        return topic, did, warnings

    # ---------- Group detection ----------
    @staticmethod
    def _word_tokens(q: str) -> List[str]:
        return re.findall(r"[a-z0-9%]+", (q or "").lower())

    @staticmethod
    def _has_any(hay: str, needles: List[str]) -> bool:
        h = hay.lower()
        return any(n in h for n in needles)

    @staticmethod
    def _contains_any_token(tokens: List[str], vocab: List[str]) -> bool:
        s = set(tokens)
        return any(v in s for v in vocab)

    def _available_groups(self, topic_df: pd.DataFrame) -> List[str]:
        return [str(g) for g in topic_df["group"].dropna().unique()]

    def _first_available_group(self, available: List[str], preds: List[str]) -> Optional[str]:
        # return the first available group whose LOWER name starts with or contains any of preds
        al = [a.lower() for a in available]
        for pref in preds:
            for i, gl in enumerate(al):
                if gl.startswith(pref) or pref in gl:
                    return available[i]
        return None

    def _orientation_tokens_present(self, tokens: List[str]) -> bool:
        orient_vocab = {
            "gay","lesbian","bisexual","straight","homosexual","lgb","lgbt","lgbtq","orientation","identity"
        }
        return any(t in orient_vocab for t in tokens)

    def _groups_signaled(self, query: str, topic_df: pd.DataFrame) -> List[str]:
        """
        Robust group signals:
          - Sexual orientation/identity > Sex
          - Region, Poverty/Income, Race/Hispanic, Insurance
          - Only returns groups that actually exist in the dataset
          - If orientation is requested but not available in the dataset → return [] (TOTAL fallback later)
        """
        q = self.clean_text(query)
        tokens = self._word_tokens(q)
        available = self._available_groups(topic_df)
        hits: List[str] = []

        # 1) Hints from mappings (kept, but we will still apply precedence rules)
        try:
            map_hits, _ = self._infer_group_and_sub_from_mappings(q, topic_df)
        except Exception:
            map_hits = []

        # 2) Sexual orientation/identity takes precedence
        have_orient_group = self._first_available_group(
            available, ["sexual orientation", "sexual identity"]
        )
        orientation_requested = self._orientation_tokens_present(tokens) or \
                                ("sexual orientation" in q.lower()) or ("sexual identity" in q.lower())

        if orientation_requested:
            if have_orient_group:
                hits.append(have_orient_group)
            else:
                # Do NOT fall back to Sex if orientation requested but absent
                # (Totals path will handle the exact warning string.)
                return []  # force totals
        else:
            # 3) Sex only if orientation wasn't requested
            if any(t in {"male","men","female","women","sex","gender","man","woman"} for t in tokens):
                g = self._first_available_group(available, ["sex"])
                if g:
                    hits.append(g)

        # 4) Region
        if any(t in {"midwest","north","northeast","south","west","mid-west"} for t in tokens) or "region" in q.lower():
            g = self._first_available_group(available, ["region"])
            if g:
                hits.append(g)

        # 5) Poverty / Income (FPL etc.)
        if any(t in {"poor","poverty","fpl","income","low","low-income","lowincome"} for t in tokens) or \
           self._has_any(q, ["% fpl", "income-to-poverty", "poverty ratio", "poverty status"]):
            g = self._first_available_group(available, ["poverty", "income-to-poverty", "poverty ratio", "fpl", "income ratio"])
            if g:
                hits.append(g)

        # 6) Race & Hispanic
        if any(t in {"black","white","asian","hispanic","latino","latina","latinx","race","ethnic","ethnicity"} for t in tokens):
            g = self._first_available_group(available, ["race & hispanic", "race/ethnicity", "race and hispanic", "race"])
            if g:
                hits.append(g)

        # 7) Health insurance
        if any(t in {"insurance","medicaid","medicare","uninsured","private","coverage"} for t in tokens):
            g = self._first_available_group(available, ["health insurance coverage","insurance"])
            if g:
                hits.append(g)

        # Merge with mapping hits but keep orientation/sex precedence we already applied
        for h in map_hits:
            if h not in hits and h in available:
                hits.append(h)

        # Dedup, preserve order
        seen = set()
        out = []
        for h in hits:
            if h not in seen:
                out.append(h); seen.add(h)
        return out

    def _strong_group_signal(self, query: str, topic_df: pd.DataFrame) -> Optional[str]:
        if self.strict_no_implicit_group:
            return None
        hits = self._groups_signaled(query, topic_df)
        return hits[0] if hits else None

    # ---------- Requested subgroup extraction ----------
    def _requested_subgroups_for_group(self, query: str, group_label: str, subgroups: List[str]) -> List[str]:
        """
        Given a group and its available subgroups, return the list of subgroups
        explicitly requested in the query (by synonyms or substrings), preserving
        request order; only returns labels that exist in `subgroups`.
        """
        q = (query or "").lower()
        toks = self._word_tokens(q)
        subs_l = [s.lower() for s in subgroups]
        req: List[str] = []

        def _push_if(match_fn):
            for i, s in enumerate(subs_l):
                if match_fn(s):
                    lab = subgroups[i]
                    if lab not in req:
                        req.append(lab)

        canon = self.canonicalize_group(group_label).lower()

        if canon.startswith("sexual orientation") or canon.startswith("sexual identity"):
            # Bisexual
            if "bisexual" in toks or "bi" in toks:
                _push_if(lambda s: "bisexual" in s)
            # Gay/Lesbian/Homosexual
            if any(t in toks for t in ["gay","lesbian","homosexual","lgb","lgbt","lgbtq"]):
                _push_if(lambda s: ("gay" in s) or ("lesbian" in s) or ("homosexual" in s))
            # Straight/Heterosexual
            if ("straight" in toks) or self._has_any(q, ["heterosexual"]):
                _push_if(lambda s: "straight" in s or "heterosexual" in s)

        elif canon == "sex":
            if any(t in toks for t in ["women","woman","female"]):
                _push_if(lambda s: "female" in s)
            if any(t in toks for t in ["men","man","male"]):
                _push_if(lambda s: "male" in s)

        elif "region" in canon:
            if "midwest" in toks or "mid-west" in q:
                _push_if(lambda s: "midwest" in s)
            if "northeast" in toks or "north" in toks and "east" in toks:
                _push_if(lambda s: "northeast" in s or "north east" in s)
            if "south" in toks:
                _push_if(lambda s: "south" in s)
            if "west" in toks:
                _push_if(lambda s: "west" in s)

        elif ("poverty" in canon) or ("fpl" in canon) or ("income" in canon) or ("ratio" in canon):
            # "poor/low income" → prefer the lowest FPL bin (e.g., <100% FPL)
            wants_poor = any(t in toks for t in ["poor","poverty","low","low-income","lowincome","under","below"]) or \
                         self._has_any(q, ["<100", "less than 100", "0–99", "0-99", "0 to 99"])
            wants_mid  = self._has_any(q, ["100–199", "100-199", "100 to 199"])
            wants_hi   = self._has_any(q, [">=200", "≥200", "200+", "200 or more", "200 and over", "at least 200"])

            # Helper to rank FPL bins
            def fpl_rank(s: str) -> int:
                s = s.replace("\u2013","-").replace("\u2265",">=").lower()
                if ("<100" in s) or ("less than 100" in s) or ("0-99" in s) or ("0 to 99" in s):
                    return 0
                if ("100-199" in s) or ("100 to 199" in s):
                    return 1
                if (">=200" in s) or ("200+" in s) or ("200 or more" in s) or ("200 and over" in s):
                    return 2
                # Fallback keyword based heuristic
                if "less" in s or "below" in s:
                    return 0
                if "at least" in s or "200" in s:
                    return 2
                return 1

            if wants_poor:
                # choose the poorest label(s)
                min_rank = min(fpl_rank(s) for s in subs_l)
                _push_if(lambda s: fpl_rank(s) == min_rank)
            if wants_mid:
                _push_if(lambda s: ("100-199" in s) or ("100 to 199" in s) or ("100–199" in s))
            if wants_hi:
                _push_if(lambda s: (">=200" in s) or ("200+" in s) or ("200 or more" in s) or ("200 and over" in s))

        elif ("race" in canon) or ("hispanic" in canon):
            if "black" in toks or self._has_any(q, ["african american","african-american"]):
                _push_if(lambda s: ("black" in s))
            if "white" in toks or "caucasian" in toks:
                _push_if(lambda s: ("white" in s))
            if "asian" in toks:
                _push_if(lambda s: ("asian" in s))
            if any(t in toks for t in ["hispanic","latino","latina","latinx"]):
                _push_if(lambda s: ("hispanic" in s) or ("latino" in s) or ("latina" in s) or ("latinx" in s))

        elif ("insurance" in canon):
            if "uninsured" in toks:
                _push_if(lambda s: "uninsured" in s)
            if "medicaid" in toks:
                _push_if(lambda s: "medicaid" in s)
            if "medicare" in toks:
                _push_if(lambda s: "medicare" in s)
            if "private" in toks:
                _push_if(lambda s: "private" in s)

        # Final dedup (preserving order)
        out: List[str] = []
        seen = set()
        for s in req:
            if s not in seen:
                out.append(s); seen.add(s)
        return out

    def _extract_requested_subgroups(self, query: str, subgroups: List[str]) -> List[str]:
        # Back-compat shim used elsewhere; without the group label it can only do simple substring matches.
        q = self.clean_text(query).lower()
        wanted = []
        aliases = {
            "male":"male","men":"male","female":"female","women":"female",
            "hispanic":"hispanic","white":"white","black":"black","asian":"asian",
            "bisexual":"bisexual","gay":"gay","lesbian":"lesbian","straight":"straight",
            "midwest":"midwest","south":"south","west":"west","northeast":"northeast",
            "uninsured":"uninsured","medicaid":"medicaid","medicare":"medicare","private":"private",
        }
        for sg in subgroups:
            sgl = str(sg).lower()
            if sgl in q:
                wanted.append(sg)
                continue
            for a in aliases:
                if a in q and a in sgl:
                    wanted.append(sg)
                    break
        # dedup preserve
        out = []
        for w in wanted:
            if w not in out:
                out.append(w)
        return out

    def _order_subgroups(self, group_label: str, subs: List[str], query: Optional[str]=None) -> List[str]:
        # Named subgroup(s) first (group-aware), then any configured order, then remaining alphabetical.
        requested = self._requested_subgroups_for_group(query or "", group_label, subs) if query else []
        gkey = self.canonicalize_group(group_label)
        order_map = self.group_label_map.get("orders", {}).get(gkey) or self.group_label_map.get("orders", {}).get(group_label)
        ordered: List[str] = []

        # 1) Requested first
        for r in requested:
            if r in subs and r not in ordered:
                ordered.append(r)
        # 2) Configured canonical order
        if order_map:
            for s in order_map:
                if s in subs and s not in ordered:
                    ordered.append(s)
        # 3) Remaining alphabetical
        for s in sorted(subs):
            if s not in ordered:
                ordered.append(s)
        return ordered

    # ---------- Period selection ----------
    def extract_year_info(self, query: str):
        ql = query.lower()
        years = sorted({int(y) for y in re.findall(r'\b(20\d{2}|19[89]\d)\b', ql)})
        want_latest  = any(k in ql for k in ['latest','most recent','current'])
        want_lastyr  = 'last year' in ql
        want_range   = any(k in ql for k in ['all years','time series','over time','across all years','full range'])
        return years, want_latest, want_lastyr, want_range

    def get_best_periods(self, query: str, periods_df: pd.DataFrame):
        """Return sorted list of period_label values (use ALL by default if request is unconstrained)."""
        if periods_df.empty:
            return [], "YR000: No data available"

        df = periods_df.dropna(subset=["period_label"]).drop_duplicates(subset=["period_label","period_start","period_end"]).copy()
        df = df.sort_values(["period_start","period_end"], na_position="last")
        labels = df["period_label"].tolist()
        starts = df["period_start"].tolist()
        ends   = df["period_end"].tolist()

        years, want_latest, want_lastyr, _ = self.extract_year_info(query)

        # Specific year(s) → include overlapping periods
        if years:
            y_min, y_max = min(years), max(years)
            sel = [lbl for lbl, s, e in zip(labels, starts, ends)
                   if (s is not None and e is not None and not (e < y_min or s > y_max))]
            if sel:
                return sel, ""
            # none overlapped → most recent period
            idx = max(range(len(labels)), key=lambda i: (ends[i] or -1))
            return [labels[idx]], "YR003: Data for the requested years is not available; the most recent period is shown."

        # "last year" → period that contains last year
        if want_lastyr:
            ly = datetime.now().year - 1
            sel = [lbl for lbl, s, e in zip(labels, starts, ends)
                   if (s is not None and e is not None and s <= ly <= e)]
            if sel:
                return sel, ""
            idx = max(range(len(labels)), key=lambda i: (ends[i] or -1))
            return [labels[idx]], "YR004: The last calendar year is not available; the most recent period is shown."

        # "latest" → greatest end year
        if want_latest:
            idx = max(range(len(labels)), key=lambda i: (ends[i] or -1))
            return [labels[idx]], ""

        # Default → ALL periods
        return labels, "YR005: All available periods are shown."

    # ---------- Formatting ----------
    def _round(self, x, inc=0.1):
        try:
            return round(float(x) / inc) * inc
        except Exception:
            return x

    def fmt_pct(self, est, lci, uci, inc=0.1):
        if est is None:
            return None
        e = self._round(est, inc)
        if lci is None or uci is None:
            return f"{e:.1f}%"
        return f"{e:.1f}% (95% CI: {self._round(lci,inc):.1f}-{self._round(uci,inc):.1f}%)"

    def safe_float(self, v):
        try:
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    # ---------- Period-based lookups ----------
    def _hi_lo_for_period(self, gdf: pd.DataFrame, period_label: str):
        yd = gdf[gdf['period_label'] == period_label]
        if yd.empty:
            return None
        rows = []
        for _,row in yd.iterrows():
            sg = str(row.get("subgroup"))
            est = self.safe_float(row.get("estimate"))
            lci = self.safe_float(row.get("estimate_lci")); uci = self.safe_float(row.get("estimate_uci"))
            if est is not None and 'total' not in sg.lower():
                rows.append((sg,est,lci,uci))
        if not rows:
            return None
        hi = max(rows, key=lambda r: r[1]); lo = min(rows, key=lambda r: r[1])
        return hi, lo

    def _format_point(self, tup):
        if not tup:
            return ""
        sg, e, l, u = tup
        if e is None:
            return f"{sg}: n/a"
        if l is not None and u is not None:
            return f"{sg}: {e:.1f}% (95% CI: {l:.1f}-{u:.1f}%)"
        return f"{sg}: {e:.1f}%"

    def subgroup_series_periods(self, df: pd.DataFrame, ticks: List[str], subgroups: List[str]):
        results = []
        for subgroup in subgroups:
            sdata = df[df['subgroup'] == subgroup]
            series = []
            for t in ticks:
                yd = sdata[sdata['period_label'] == t]
                if not yd.empty:
                    row = yd.iloc[0]
                    series.append({
                        "period": t,
                        "estimate": self.safe_float(row.get("estimate")),
                        "ci_lower": self.safe_float(row.get("estimate_lci")),
                        "ci_upper": self.safe_float(row.get("estimate_uci")),
                    })
            if series:
                results.append({"subgroup": subgroup, "series": series})
        return results

    # ---------- Grouped summary ----------
    def _subgroup_hint_for_group(self, query: str, group_label: str, subgroups: List[str]) -> Optional[str]:
        """Return the first requested subgroup for a given group, or None."""
        req = self._requested_subgroups_for_group(query, group_label, subgroups)
        return req[0] if req else None

    def generate_demographic_results(self, topic_df: pd.DataFrame, target_group: str, ticks: List[str], topic: str, clean_query: str, dataset_id: str, warnings: List[str]) -> QueryResult:
        gdf = topic_df[topic_df['group'] == target_group]
        if gdf.empty:
            return QueryResult(clean_query, topic, dataset_id, target_group, [], [], [],
                               f"This breakdown ({target_group}) is not available in the public DQS tables for “{topic}”. Additional information is available at https://www.cdc.gov.",
                               0.5, warnings)

        subgroups = [sg for sg in gdf['subgroup'].unique() if sg and 'total' not in str(sg).lower()]
        # Order with subgroup-first behavior baked in
        subgroups = self._order_subgroups(target_group, subgroups, query=clean_query)

        # Additional explicit move-to-front if a single clear hint exists
        rq = self._subgroup_hint_for_group(clean_query, target_group, subgroups)
        if rq and rq in subgroups:
            subgroups = [rq] + [s for s in subgroups if s != rq]

        if not subgroups:
            return QueryResult(clean_query, topic, dataset_id, target_group, [], [], [], f"No subgroup rows available for {target_group}.", 0.6, warnings)

        series = self.subgroup_series_periods(gdf, ticks, subgroups)
        if not series:
            return QueryResult(clean_query, topic, dataset_id, target_group, [], [], [], f"No estimates available for {target_group} in the selected period(s).", 0.6, warnings)

        header = f"{topic} by {self.canonicalize_group(target_group)}"
        notes: List[str] = []

        multi_groups = self._groups_signaled(clean_query, topic_df)
        if len(multi_groups) > 1:
            keep = self.canonicalize_group(target_group)
            notes.append(f"Multi-dimensional cross-tabulations are not supported in the public DQS tables; results are shown by {keep}.")

        requested = self._requested_subgroups_for_group(clean_query, target_group, subgroups)
        if requested:
            notes.append("Named subgroup(s) appear first; additional subgroups are provided for context.")

        if len(ticks) == 1:
            t = ticks[0]
            hilow = self._hi_lo_for_period(gdf, t)
            if hilow:
                hi, lo = hilow
                summary = f"{header} in {t}: Highest — {self._format_point(hi)}; Lowest — {self._format_point(lo)}."
            else:
                summary = f"{header} in {t}: no comparable subgroup estimates."
        else:
            first = ticks[0]; last = ticks[-1]
            first_hl = self._hi_lo_for_period(gdf, first)
            last_hl  = self._hi_lo_for_period(gdf, last)
            parts = [f"{header} ({first}–{last})."]
            if first_hl: parts.append(f"In {first}, Highest — {self._format_point(first_hl[0])}; Lowest — {self._format_point(first_hl[1])}.")
            if last_hl:  parts.append(f"In {last}, Highest — {self._format_point(last_hl[0])}; Lowest — {self._format_point(last_hl[1])}.")
            summary = " ".join(parts)

        if notes:
            summary += " " + " ".join(notes)

        if any(w.startswith("AGE101") for w in warnings):
            summary = "Estimates for children on this topic are not available in the public DQS tables. Adult estimates are provided below. " + summary

        return QueryResult(clean_query, topic, dataset_id, target_group, subgroups, [], series, summary, 0.9, warnings)

    # ---------- Totals ----------
    def _is_total_like(self, row: pd.Series) -> bool:
        cls = str(row.get("classification","")).lower()
        grp = str(row.get("group","")).lower()
        sub = str(row.get("subgroup","")).lower()
        if cls == "total" or grp == "total":
            return True
        pats = [r"^total$", r"^overall$", r"^all ages?$",
                r"(adults?\s*18.*older|18\s*years\s*(and|&)?\s*older|^18\+$)",
                r"(children.*<\s*18|under\s*18|younger\s*than\s*18|<\s*18|<=\s*17)",
                r"^all adults?$", r"^all children$"]
        return any(re.search(rx, sub) for rx in pats)

    def _preferred_total_subgroup(self, dataset_id: Optional[str], topic: str) -> Optional[str]:
        try:
            if not dataset_id:
                return None
            return self.dataset_overrides.get(dataset_id, {}).get(topic, {}).get("preferred_total_subgroup")
        except Exception:
            return None

    def _pick_total_row_for_period(self, topic_df: pd.DataFrame, period_label: str)->Optional[pd.Series]:
        ydf = topic_df[topic_df["period_label"] == period_label]
        if ydf.empty:
            return None
        dsid = ydf.get("dataset_id", pd.Series([None])).dropna().iloc[0] if "dataset_id" in ydf.columns and not ydf.empty else None
        tstr = ydf.get("topic", pd.Series([None])).dropna().iloc[0] if "topic" in ydf.columns and not ydf.empty else ""
        pref = self._preferred_total_subgroup(dsid, tstr)
        if pref:
            cand = ydf[ydf["subgroup"].astype(str) == pref]
            if len(cand):
                return cand.iloc[0]
        cands = ydf[ydf.apply(self._is_total_like, axis=1)]
        if len(cands):
            return cands.iloc[0]
        return ydf.iloc[0]

    def _totals_period_series(self, topic_df: pd.DataFrame, ticks: List[str]):
        points = []
        for t in ticks:
            row = self._pick_total_row_for_period(topic_df, t)
            if row is None:
                continue
            est = self.safe_float(row.get("estimate")); l = self.safe_float(row.get("estimate_lci")); u = self.safe_float(row.get("estimate_uci"))
            points.append((t, est, l, u))
        return points

    def generate_total_results(self, topic_df: pd.DataFrame, ticks: List[str], topic: str, clean_query: str, dataset_id: str, warnings: List[str]) -> QueryResult:
        # Dataset-aware orientation → TOTAL fallback (exact warning string)
        q = (clean_query or "").lower()
        tokens = self._word_tokens(q)
        has_orientation_kw = self._orientation_tokens_present(tokens) or \
                             ("sexual orientation" in q) or ("sexual identity" in q)
        if has_orientation_kw:
            available = [str(g).lower() for g in topic_df["group"].dropna().unique()]
            orient_present = any(gl.startswith("sexual orientation") or gl.startswith("sexual identity") for gl in available)
            if not orient_present:
                warnings.append("Sexual orientation is not available in this dataset; showing total instead.")

        if not ticks:
            return QueryResult(clean_query, topic, dataset_id, None, [], [], [],
                               f"An overall estimate for “{topic}” is not available in the Data Query System (DQS). Additional information is available at https://www.cdc.gov.",
                               0.5, warnings)

        ts = self._totals_period_series(topic_df, ticks)
        if not ts:
            return QueryResult(clean_query, topic, dataset_id, None, [], [], [],
                               f"No overall estimates for the selected period(s).", 0.6, warnings)

        first = ticks[0]; last = ticks[-1]
        by_tick = {t:(e,l,u) for (t,e,l,u) in ts if e is not None}
        latest = by_tick.get(last); earliest = by_tick.get(first)
        stats = [(t,e) for t,(e,_,_) in by_tick.items() if e is not None]
        if stats:
            hi_t, hi_v = max(stats, key=lambda t: t[1])
            lo_t, lo_v = min(stats, key=lambda t: t[1])
        else:
            hi_t=lo_t=hi_v=lo_v=None

        parts = [f"{topic} — overall"]
        if first == last and latest:
            e,l,u = latest; parts.append(f"in {last}: {self.fmt_pct(e,l,u)}.")
        else:
            if earliest:
                e1,l1,u1 = earliest; parts.append(f"in {first}: {self.fmt_pct(e1,l1,u1)};")
            if latest:
                e2,l2,u2 = latest; parts.append(f"in {last}: {self.fmt_pct(e2,l2,u2)}.")
            if hi_t is not None and lo_t is not None and hi_t != lo_t:
                parts.append(f"Across {first}–{last}, values ranged from {lo_v:.1f}% ({lo_t}) to {hi_v:.1f}% ({hi_t}).")
        summary = " ".join(parts)

        if any(w.startswith("AGE101") for w in warnings):
            summary = "Estimates for children on this topic are not available in the public DQS tables. Adult estimates are provided below. " + summary

        return QueryResult(clean_query, topic, dataset_id, None, [], [], [], summary, 0.85, warnings)

    # ---------- Main ----------
    def extract_composite_group_from_query(self, query: str, available_groups: List[str]) -> Optional[str]:
        ql = query.lower()
        for g in sorted(available_groups, key=len, reverse=True):
            gl = g.lower()
            if gl == 'total':
                continue
            gl_clean = re.sub(r'[:\(\),]','',gl).strip()
            for pat in (rf'(?<!\w)by\s+{re.escape(gl)}(?!\w)', rf'(?<!\w)by\s+{re.escape(gl_clean)}(?!\w)'):
                if re.search(pat, ql):
                    return g
        return None

    def query(self, query_text: str) -> QueryResult:
        if self.master_df is None:
            return QueryResult(query_text, None, None, None, [], [], [], "Master data is not available for this request.", 0.0, ["MD000: Master data not available"])

        clean_query = self.clean_text(query_text)
        warnings: List[str] = []

        topic, dataset_id, tw = self.match_topic_adult_first(clean_query)
        warnings.extend(tw)
        if not topic:
            msg = ("Data for this topic is not currently available in the Data Query System (DQS). "
                   "Additional information is available at https://www.cdc.gov.")
            return QueryResult(clean_query, None, None, None, [], [], [], msg, 0.1, warnings)

        topic_df = self.master_df[self.master_df['topic'] == topic].copy()
        if topic_df.empty:
            return QueryResult(clean_query, topic, dataset_id, None, [], [], [], f"Public tables for “{topic}” are not available in the Data Query System (DQS). Additional information is available at https://www.cdc.gov.", 0.5, warnings)

        # Build periods
        periods_df = topic_df[["period_label","period_start","period_end"]].copy()
        ticks, expl = self.get_best_periods(clean_query, periods_df)
        if expl:
            warnings.append(expl)
        if not ticks and not periods_df.empty:
            last_tick = str(periods_df.sort_values(["period_end"]).iloc[-1]["period_label"])
            ticks = [last_tick]

        has_by = bool(re.search(r'(?<!\w)by(?!\w)', clean_query, re.I))
        strong_group = self._strong_group_signal(clean_query, topic_df)

        if has_by or strong_group:
            target_group = strong_group
            if not target_group:
                g = self.extract_composite_group_from_query(clean_query, list(topic_df['group'].unique()))
                target_group = g
            if target_group:
                dsid = dataset_id or (topic_df.get("dataset_id", pd.Series([None])).dropna().iloc[0] if "dataset_id" in topic_df.columns and not topic_df.empty else None)
                return self.generate_demographic_results(topic_df, target_group, ticks, topic, clean_query, dsid, warnings)

            # No group resolved → totals
            row = self._pick_total_row_for_period(topic_df, ticks[-1])
            dsid = dataset_id or (row.get("dataset_id") if row is not None else None)
            return self.generate_total_results(topic_df, ticks, topic, clean_query, dsid, warnings)

        # No "by" → totals
        dsid = dataset_id or (topic_df.get("dataset_id", pd.Series([None])).dropna().iloc[0] if "dataset_id" in topic_df.columns and not topic_df.empty else None)
        return self.generate_total_results(topic_df, ticks, topic, clean_query, dsid, warnings)


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse, sys, json as _json
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="standardized_dqs_full.json")
    p.add_argument("q", nargs="+")
    args = p.parse_args()
    eng = DeterministicQueryEngine(args.data)
    res = eng.query(" ".join(args.q))
    print(_json.dumps({
        "engine_version": ENGINE_VERSION,
        "summary": res.summary,
        "topic": res.topic,
        "dataset_id": res.dataset_id,
        "group": res.group,
        "years": res.years,
        "warnings": res.warnings
    }, indent=2))
