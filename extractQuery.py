from pathlib import Path
import re
import pandas as pd

# -----------------------------
# 1) CONFIG
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent 
LOINC_DIR = PROJECT_ROOT / "Loinc_2.82"

OUTPUT_DIR = LOINC_DIR / "GeneratedCSVs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAPPING_TXT_PATH = OUTPUT_DIR / "query_mapping_new.txt"

QUERIES = [
    "carbon dioxide in serum",
    "blood urea in serum",
    "blood urea in urine",
    "creatinine in plasma",
    "urea nitrogen in urine",
    "ammonia in plasma",
    "ammonia in serum",
    "lactate dehydrogenase in serum",
    "lactate dehydrogenase in plasma",
    "bilirubin direct in plasma",
    "bilirubin indirect in plasma",
    "bilirubin total in serum",
    "bilirubin direct in serum",
    "bilirubin indirect in serum",
    "total bile acids in serum",
    "glutamate dehydrogenase in serum",
    "cholinesterase in serum",
    "total cholesterol in plasma",
    "hdl cholesterol in plasma",
    "ldl cholesterol in plasma",
    "triglycerides in plasma",
    "lipoprotein phospholipase a2 in serum",
    "homocysteine in plasma",
    "homocysteine in serum",
    "beta hydroxybutyrate in serum",
    "beta hydroxybutyrate in plasma",
    "ketone bodies in serum",
    "free fatty acids in serum",
    "lactic acid in serum",
    "pyruvate in plasma",
    "pyruvate in serum",
    "fructosamine in serum",
    "fructosamine in plasma",
    "insulin like growth factor 1 in serum",
    "growth hormone in serum",
    "parathyroid hormone in serum",
    "calcitonin in serum",
    "gastrin in serum",
    "vitamin c in plasma",
    "vitamin c in serum",
    "thiamine in whole blood",
    "pyridoxal phosphate in plasma",
    "riboflavin in plasma",
    "niacin in plasma",
    "biotin in serum",
    "vitamin k in plasma",
    "vitamin k in serum",
    "transferrin in serum",
    "transferrin in plasma",
    "unsaturated iron binding capacity in serum",
    "unsaturated iron binding capacity in plasma",
    "hepcidin in serum",
    "hepcidin in plasma",
    "soluble transferrin receptor in serum",
    "soluble transferrin receptor in plasma",
    "retinol binding protein in serum",
    "prealbumin in serum",
    "antinuclear antibody in serum",
    "rheumatoid factor in serum",
    "anti ccp antibody in serum",
    "anti double stranded dna antibody in serum",
    "anti smith antibody in serum",
    "anti ssa antibody in serum",
    "anti ssb antibody in serum",
    "anti scl 70 antibody in serum",
    "anti centromere antibody in serum",
    "anti jo 1 antibody in serum",
    "anti mitochondria antibody in serum",
    "anti smooth muscle antibody in serum",
    "anti cardiolipin igg antibody in serum",
    "anti cardiolipin igm antibody in serum",
    "beta 2 glycoprotein 1 igg antibody in serum",
    "beta 2 glycoprotein 1 igm antibody in serum",
    "complement c3 in serum",
    "complement c4 in serum",
    "immunoglobulin g in serum",
    "immunoglobulin a in serum",
    "immunoglobulin m in serum",
    "immunoglobulin e in serum",
    "dhea sulfate in serum",
    "17 hydroxyprogesterone in serum",
    "androstenedione in serum",
    "free testosterone in serum",
    "estrone in serum",
    "anti mullerian hormone in serum",
    "parathyroid hormone related peptide in plasma",
    "insulin antibody in serum",
    "insulin receptor antibody in serum",
    "growth hormone releasing hormone in plasma",
    "vasopressin in plasma",
    "copeptin in plasma",
    "adrenocorticotropic hormone in serum",
    "metanephrine in plasma",
    "normetanephrine in plasma",
    "catecholamines in urine",
    "vanillylmandelic acid in urine",
    "homovanillic acid in urine",
    "5 hydroxyindoleacetic acid in urine",
    "reticulocytes count in blood",
    "reticulocyte hemoglobin in blood",
    "immature granulocytes count in blood",
    "band neutrophils count in blood",
    "nucleated red blood cells count in blood",
    "mean platelet volume in blood",
    "platelet distribution width in blood",
    "absolute neutrophil count in blood",
    "absolute lymphocyte count in blood",
    "absolute monocyte count in blood",
    "absolute eosinophil count in blood",
    "absolute basophil count in blood",
    "erythropoietin in serum",
    "erythropoietin in plasma",
    "haptoglobin in plasma",
    "serum free light chains kappa",
    "serum free light chains lambda",
    "kappa lambda ratio in serum",
    "activated partial thromboplastin time in plasma",
    "fibrinogen in serum",
    "factor viii activity in plasma",
    "factor ix activity in plasma",
    "factor xi activity in plasma",
    "factor xiii activity in plasma",
    "von willebrand factor antigen in plasma",
    "von willebrand factor activity in plasma",
    "antithrombin activity in plasma",
    "protein c activity in plasma",
    "protein s activity in plasma",
    "lupus anticoagulant in plasma",
    "anti xa activity in plasma",
    "uric acid in urine",
    "oxalate in urine",
    "citrate in urine",
    "urine osmolality",
    "urine sodium excretion",
    "urine potassium excretion",
    "urine urea nitrogen excretion",
    "urine creatinine excretion",
    "albumin excretion rate in urine",
    "protein excretion rate in urine",
    "total protein in urine",
    "urine calcium excretion",
    "urine chloride",
    "urine magnesium",
    "urine phosphate",
    "urine pH",
    "urine glucose",
    "urine ketones",
    "urine bilirubin",
    "urine urobilinogen",
    "vancomycin trough in serum",
    "gentamicin trough in serum",
    "amikacin trough in serum",
    "tobramycin trough in serum",
    "digoxin in serum",
    "lithium in serum",
    "phenytoin in serum",
    "free phenytoin in serum",
    "valproic acid in serum",
    "carbamazepine in serum",
    "phenobarbital in serum",
    "theophylline in serum",
    "acetaminophen in serum",
    "salicylate in serum",
    "ethanol in serum",
    "methotrexate in serum",
    "tacrolimus in whole blood",
    "cyclosporine in whole blood",
    "sirolimus in whole blood",
    "everolimus in whole blood",
    "epstein barr virus capsid igm antibody in serum",
    "epstein barr virus capsid igg antibody in serum",
    "cytomegalovirus igm antibody in serum",
    "cytomegalovirus igg antibody in serum",
    "toxoplasma gondii igm antibody in serum",
    "toxoplasma gondii igg antibody in serum",
    "rubella igg antibody in serum",
    "measles igg antibody in serum",
    "mumps igg antibody in serum",
    "varicella zoster virus igg antibody in serum",
    "chlamydia trachomatis rna in urine",
    "neisseria gonorrhoeae rna in urine",
    "group a streptococcus antigen in throat swab",
    "respiratory syncytial virus rna in nasopharyngeal swab",
    "adenovirus rna in nasopharyngeal swab",
    "mycoplasma pneumoniae rna in nasopharyngeal swab",
    "clostridioides difficile toxin b gene in stool",
    "rotavirus antigen in stool",
    "norovirus rna in stool",
    "campylobacter jejuni rna in stool",
    "total igE in serum",
    "dust mite igE antibody in serum",
    "cat dander igE antibody in serum",
    "dog dander igE antibody in serum",
    "birch pollen igE antibody in serum",
    "grass pollen igE antibody in serum",
    "peanut igE antibody in serum",
    "egg white igE antibody in serum",
    "milk igE antibody in serum",
    "wheat igE antibody in serum",
    "lactate in cerebrospinal fluid",
    "oligoclonal bands in cerebrospinal fluid",
    "immunoglobulin g index in cerebrospinal fluid",
    "albumin in cerebrospinal fluid",
    "opening pressure in cerebrospinal fluid",
    "protein electrophoresis in serum",
    "immunofixation in serum",
    "fecal fat in stool",
    "pancreatic elastase in stool",
    "lactoferrin in stool",
    "giardia antigen in stool",
    "cryptosporidium antigen in stool",
    "ova and parasites in stool",
    "stool culture",
    "shiga toxin in stool",
    "salmonella rna in stool",
    "shigella rna in stool",
    "basic metabolic panel in serum",
    "comprehensive metabolic panel in serum",
    "hepatic function panel in serum",
    "lipid panel in serum",
    "thyroid function panel in serum",
    "iron studies panel in serum",
    "coagulation panel in plasma",
    "urinalysis panel in urine",
]

MIN_ROWS_TO_EXPORT = 5

TEXT_COLUMNS_CANDIDATES = [
    "LONG_COMMON_NAME",
    "SHORTNAME",
    "COMPONENT",
    "SYSTEM",
    "PROPERTY",
    "CLASS",
    "METHOD_TYP",
    "STATUS_TEXT",
]

def slugify_query(q: str) -> str:
    # create a filename-friendly slug
    slug = q.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug

def build_contains_pattern(q: str) -> re.Pattern:
    """
    Builds a flexible regex that matches query tokens in order, allowing words in-between.
    Example: "glucose in blood" -> r"\bglucose\b.*\bin\b.*\bblood\b"
    """
    tokens = [t for t in re.split(r"\s+", q.strip()) if t]
    parts = [rf"\b{re.escape(t)}\b" for t in tokens]
    pattern = ".*".join(parts)
    return re.compile(pattern, re.IGNORECASE)

def autodetect_loinc_csv(loinc_dir: Path) -> Path:
    candidates = list(loinc_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No .csv found under: {loinc_dir}")

    # Prefer a large "Loinc.csv" or "LoincTable.csv"-like file
    preferred = sorted(
        candidates,
        key=lambda p: (
            0 if p.name.lower() in {"loinc.csv", "loinccsv.csv", "loincready.csv", "loinccore.csv", "loinctable.csv"} else 1,
            -p.stat().st_size
        )
    )
    return preferred[0]

if LOINC_TABLE_PATH is None:
    LOINC_TABLE_PATH = autodetect_loinc_csv(LOINC_DIR)
    print(f"[INFO] Auto-detected LOINC table CSV: {LOINC_TABLE_PATH}")

loinc = pd.read_csv(LOINC_TABLE_PATH, low_memory=False)

TEXT_COLUMNS = [c for c in TEXT_COLUMNS_CANDIDATES if c in loinc.columns]
if not TEXT_COLUMNS:
    raise ValueError(
        "None of the expected text columns were found in the LOINC table. "
        f"Available columns include: {list(loinc.columns)[:30]}..."
    )

for c in TEXT_COLUMNS:
    loinc[c] = loinc[c].fillna("").astype(str)

mapping_lines = []
exported = 0
skipped = 0

for q in QUERIES:
    pattern = build_contains_pattern(q)

    mask = False
    for col in TEXT_COLUMNS:
        mask = mask | loinc[col].str.contains(pattern)

    subset = loinc[mask].copy()
    n = len(subset)

    slug = slugify_query(q)
    out_name = f"Loinc_2.82_{slug}.csv"
    out_path = OUTPUT_DIR / out_name

    if n >= MIN_ROWS_TO_EXPORT:
        subset.to_csv(out_path, index=False)
        mapping_lines.append(f'"{out_name}": "{q}",')
        exported += 1
        print(f"[OK] {q} -> {n} rows -> {out_name}")
    else:
        skipped += 1
        print(f"[SKIP] {q} -> only {n} rows (min={MIN_ROWS_TO_EXPORT})")

with open(MAPPING_TXT_PATH, "w", encoding="utf-8") as f:
    f.write("query_mapping_additions = {\n")
    for line in mapping_lines:
        f.write(f"    {line}\n")
    f.write("}\n")

print("\n--- Summary ---")
print(f"Exported CSVs: {exported}")
print(f"Skipped: {skipped}")
print(f"Mapping file written to: {MAPPING_TXT_PATH}")
