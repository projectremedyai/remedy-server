#!/usr/bin/env python3
"""Reclassify the 33 LAMC font-residue files by REAL veraPDF rule + font identity.

Read-only. Produces a per-file routing table so the deterministic passes
(width rewrite, cmap-inversion ToUnicode) can run before any OCR.
"""
import subprocess, sys, os, re, io, json
import xml.etree.ElementTree as ET
import pikepdf
from fontTools.ttLib import TTFont

PDIR = os.path.expanduser("~/code/lamc_district_forms/lamc_remediated/remediated_pdfs")

# (filename, clauses-per-handoff-appendix) — clauses only used as a cross-check
TARGETS = [
 "2022-LAMC-Institution-Set-Standards-Data.pdf","2023-2024-LAMC-Scholarship-Application.pdf",
 "2023_winter_schedule_01.12.23.pdf","2024 Winter Schedule of Classes.pdf","2025 Winter Schedule.pdf",
 "2026 Winter Schedule.pdf","238-Entry-Skills.pdf","240entryskills.pdf",
 "Biotechnology-Exhibition-Spring-2022.pdf","Certificate-Mail-form-2.pdf",
 "Follow-Up Report ACCJC 05.13.14.pdf","High School Graduation Update Form.pdf",
 "LA-Mission-CDC-Parent-Handbook-Revised-8-18.pdf","Mentorship-Prog-App.pdf",
 "NEW-LACCD-Unlawful-Discrimination-Complaint-Form.pdf","Nursing Grid.pdf","ParkingBrochure07.pdf",
 "Student-Contract.pdf","Summer 2013 (6.20).pdf","Summer2015(5-27-15).pdf",
 "Supplemental-Residency-Questionnaire-2018_2019-FILLABLE.pdf","Supplemental_Residency_Questionnaire.pdf",
 "VERIFICATION OF ENROLLMENT FORM- (FILLABLE).pdf","Winter2016schedule.pdf","getcoursefile (1).pdf",
 "math238entryskillsAnswers.pdf","summer05_schedule.pdf","spring06_schedule.pdf","winte06_schedule.pdf",
 "2020-Summer-Schedule-of-Classes-07-12-20.pdf","2020-Winter-Schedule-01-10-2020.pdf",
 "2021-Summer-Schedule-06-25-21.pdf","2022-Winter-Schedule-of-Classes-01-04-22.pdf",
]

SYMBOL_HINTS = ("wingding","dingbat","symbol","zapf","cmsy","cmmi","cmex","marlett","webding",
                "mingliu","simsun","mincho","gothic","batang","gulim","mssong","stsong","stkaiti")

def strip_ns(t): return re.sub(r"\{.*?\}","",t)
def norm_base(b):
    b = str(b).lstrip("/")
    if "+" in b[:8]: b = b.split("+",1)[1]           # drop subset tag ABCDEE+
    return b
def is_symbolish(name):
    n = name.lower()
    return any(h in n for h in SYMBOL_HINTS)

# ---------- 1. run veraPDF once over all present files ----------
present = [f for f in TARGETS if os.path.isfile(os.path.join(PDIR,f))]
missing = [f for f in TARGETS if f not in present]
paths = [os.path.join(PDIR,f) for f in present]
print(f"# veraPDF sweep: {len(present)} present, {len(missing)} missing", file=sys.stderr)
if missing: print("MISSING:", missing, file=sys.stderr)

proc = subprocess.run(["verapdf","-f","ua1","--format","xml",*paths],
                      capture_output=True, text=True, timeout=1200)
root = ET.fromstring(proc.stdout)

# map absolute path -> list of failed font rules {clause,test,failedChecks,contexts[]}
fails = {}
for job in root.iter():
    if strip_ns(job.tag) != "job": continue
    name = None
    for ch in job.iter():
        if strip_ns(ch.tag) == "name" and ch.text: name = ch.text.strip(); break
    if not name: continue
    rules = []
    for rule in job.iter():
        if strip_ns(rule.tag) != "rule" or rule.attrib.get("status") != "failed": continue
        clause = rule.attrib.get("clause","")
        if not clause.startswith("7.21"): continue
        ctxs = []
        for c in rule.iter():
            if strip_ns(c.tag) == "context" and c.text: ctxs.append(c.text.strip())
        rules.append({"clause":clause,"test":rule.attrib.get("testNumber",""),
                      "failedChecks":int(rule.attrib.get("failedChecks","0") or 0),"contexts":ctxs})
    fails[os.path.realpath(name)] = rules

# ---------- 2. per-file font inventory ----------
def font_inventory(path):
    """normalized basefont -> {subtypes,set embedded,has_cmap,has_tu,symbol}"""
    inv = {}
    with pikepdf.open(path) as pdf:
        for obj in pdf.objects:
            try:
                if not (isinstance(obj,pikepdf.Object) and isinstance(obj,pikepdf.Dictionary)): continue
                if obj.get("/Type") != pikepdf.Name("/Font"): continue
            except Exception:
                continue
            sub = str(obj.get("/Subtype",""))
            base = obj.get("/BaseFont")
            if base is None: continue
            nb = norm_base(base)
            rec = inv.setdefault(nb, {"subtypes":set(),"embedded":False,"has_cmap":False,
                                      "has_tu":False,"symbol":is_symbolish(nb),"progtype":set()})
            rec["subtypes"].add(sub)
            if "/ToUnicode" in obj: rec["has_tu"] = True
            # locate descriptor + program (simple or via descendant CIDFont)
            descs = []
            if sub == "/Type0":
                for d in (obj.get("/DescendantFonts") or []):
                    fd = d.get("/FontDescriptor")
                    if fd is not None: descs.append(fd)
            else:
                fd = obj.get("/FontDescriptor")
                if fd is not None: descs.append(fd)
            for fd in descs:
                prog = None; ptype=None
                for key in ("/FontFile2","/FontFile3","/FontFile"):
                    if key in fd: prog = fd[key]; ptype=key; break
                if prog is None: continue
                rec["embedded"] = True; rec["progtype"].add(ptype)
                if ptype == "/FontFile2":
                    try:
                        tt = TTFont(io.BytesIO(bytes(prog.read_bytes())))
                        if "cmap" in tt and tt.getBestCmap():
                            rec["has_cmap"] = True
                    except Exception:
                        pass
    return inv

# ---------- 3. classify ----------
def classify(clauses, inv, fails_rules):
    """return (passes:set, tier:str, notes:list)"""
    passes=set(); notes=[]
    # gather failing fonts per clause from contexts
    def fonts_for(clause_prefix):
        out=set()
        for r in fails_rules:
            if not r["clause"].startswith(clause_prefix): continue
            for ctx in r["contexts"]:
                for nb in inv:
                    if nb.lower() in ctx.lower(): out.add(nb)
        return out
    have = {r["clause"] for r in fails_rules}
    tests = {(r["clause"],r["test"]) for r in fails_rules}
    if "7.21.4.1" in have:
        passes.add("EMBED"); notes.append("font(s) not embedded → embed named font")
    if "7.21.5" in have:
        passes.add("WIDTH")
    if "7.21.8" in have:
        passes.add("NOTDEF")
    if "7.21.7" in have:
        ff = fonts_for("7.21.7")
        # decide sub-route
        sub=set()
        for nb in (ff or set()):
            rec = inv.get(nb,{})
            if rec.get("symbol"): sub.add("OCR/symbol")
            elif rec.get("has_cmap"): sub.add("CMAP_INVERT")
            elif rec.get("embedded"): sub.add("REFFONT/AGL")
            else: sub.add("EMBED+MAP")
        if ("7.21.7","2") in tests: sub.add("REPAIR(bad-values)")
        if not ff: sub.add("REPAIR/UNKNOWN(font not linked)")
        passes.add("TOUNICODE:"+"|".join(sorted(sub)))
        notes.append("ToUnicode fonts: "+(", ".join(sorted(ff)) or "unlinked"))
    # tier
    hard = ("NOTDEF" in passes) or ("EMBED" in passes) or any(
        p.startswith("TOUNICODE") and ("OCR/symbol" in p or "REFFONT" in p or "EMBED+MAP" in p) for p in passes)
    det_only = passes and not hard and all(
        p=="WIDTH" or p.startswith("TOUNICODE:") and set(p.split(":")[1].split("|")) <= {"CMAP_INVERT","REPAIR(bad-values)"}
        for p in passes)
    if det_only: tier="DETERMINISTIC"
    elif ("NOTDEF" in passes or "EMBED" in passes) and not any("OCR/symbol" in p for p in passes):
        tier="SEMI-AUTO"
    else:
        tier="OCR-TAIL"
    return passes, tier, notes

rows=[]
for f in present:
    ap = os.path.realpath(os.path.join(PDIR,f))
    fr = fails.get(ap, [])
    clause_str = ", ".join(sorted({f"{r['clause']}-{r['test']}" for r in fr})) or "(none-now-passing?)"
    try:
        inv = font_inventory(os.path.join(PDIR,f))
    except Exception as e:
        inv = {};
    passes, tier, notes = classify(None, inv, fr)
    rows.append({"file":f,"clauses":clause_str,"passes":sorted(passes),"tier":tier,"notes":notes})

# ---------- 4. output ----------
order={"DETERMINISTIC":0,"SEMI-AUTO":1,"OCR-TAIL":2}
rows.sort(key=lambda r:(order.get(r["tier"],9), r["file"]))
print("\n## Per-file routing (real veraPDF clauses)\n")
print("| Tier | File | Failing rule(s) | Recommended pass(es) |")
print("|---|---|---|---|")
for r in rows:
    print(f"| {r['tier']} | {r['file']} | {r['clauses']} | {'; '.join(r['passes'])} |")
from collections import Counter
tc=Counter(r["tier"] for r in rows)
print("\n## Tier summary\n")
for t in ("DETERMINISTIC","SEMI-AUTO","OCR-TAIL"):
    print(f"- **{t}**: {tc.get(t,0)}")
outp=os.path.join(os.path.dirname(__file__),"routing.json")
json.dump(rows, open(outp,"w"), indent=2)
print(f"\nrouting.json -> {outp}")
