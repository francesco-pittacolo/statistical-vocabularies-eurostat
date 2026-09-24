# Statistical Vocabularies in Eurostat Tables

This repository contains the implementation of the assignment **“Statistical Vocabularies in Eurostat Tables”**.

The project processes Eurostat statistical tables to extract statistical vocabularies, classify vocabulary terms, assign measures to statistical domains, and identify semantic relationships between measures.

The pipeline can be executed on either the **2,000-table subset** or the **full 7,605-table dataset**.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config.py
├── src/
│   ├── step1_extract_d_t.py
│   ├── step2_extract_s_t.py
│   ├── step3_build_geo_t_and_v_t.py
│   ├── step4_extract_titles.py
│   ├── step5_build_v_total.py
│   ├── step6_classify_category.py
│   ├── step7_domain_classification.py
│   └── step8_semantic_relationships.py
├── evaluate/
│   ├── evaluate_step_6.py
│   ├── evaluate_step_7.py
│   └── evaluate_step_8.py
├── docs/
│   ├── Assignment_STAR-5.pdf
│   └── report_fact_checking.pdf
├── data/
│   ├── eurostat_2000_tables/
│   ├── eurostat_7605_tables/
│   ├── reference/
│   └── cache/
├── prompts/
│   ├── step7_domain_classification.txt
│   ├── step8_semantic_relationships.txt
│   └── step8_llm_judge.txt
└── results/
    ├── step_outputs/
    │   ├── 2000/
    │   └── 7605/
    └── evaluation/
        ├── step6/
        ├── step7/
        └── step8/
```

Pipeline outputs are stored separately for the two datasets under `results/step_outputs/`, while manual evaluation artifacts are stored under `results/evaluation/`.

The original assignment specification is available in `docs/Assignment_STAR-5.pdf`.

The project report is available in `docs/report.pdf`.

## Installation

The project is implemented in Python.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Setup

The project can be run on either the 2,000-table subset or the full 7,605-table dataset.

The 2,000-table subset is included in this repository under:

```text
data/eurostat_2000_tables/
```

The full 7,605-table dataset is not included in this repository because of its size. It must be downloaded separately from Zenodo:

https://zenodo.org/records/15681384

After downloading and extracting the archive, place the Eurostat tables in:

```text
data/eurostat_7605_tables/
```

The pipeline does not download the datasets automatically. The selected dataset must be available locally before running the corresponding pipeline.

## Dataset Selection

Select the dataset by changing `DATASET` in `config.py`.

For the 2,000-table subset:

```python
DATASET = "2000"
```

For the full 7,605-table dataset:

```python
DATASET = "7605"
```

The corresponding input and output directories are selected automatically.

## API Configuration

Steps 7 and 8 may use the Groq API.

Create a `.env` file in the repository root:

```text
API_KEY=your_groq_api_key
```

The `.env.example` file can be used as a template.

Previously obtained LLM responses are reused from `data/cache/` when available. If all required responses are already cached, no new API request is issued.

LLM prompt templates used in Steps 7 and 8 are available under `prompts/`.

## Notation

For a Eurostat table \(t\), the pipeline uses the following notation:

- \(D(t)\): temporal information identified in table \(t\);
- \(S(t)\): textual information extracted from the descriptive part of table \(t\);
- \(Geo(t)\): geographic information identified in table \(t\);
- \(V(t)\): vocabulary extracted from table \(t\), organized by dimension or context;
- \(V\): global vocabulary obtained by combining the vocabularies of all processed tables.

The global vocabulary \(V\) is subsequently partitioned into:

- `M`: measures;
- `N`: dimension names;
- `A`: dimension values;
- `U`: units;
- `O`: other terms.

## Pipeline

The eight steps are:

1. **Step 1** — extract temporal information \(D(t)\);
2. **Step 2** — extract textual information \(S(t)\);
3. **Step 3** — identify geographic information \(Geo(t)\) and build \(V(t)\);
4. **Step 4** — enrich \(V(t)\) using table titles;
5. **Step 5** — construct the global vocabulary \(V\);
6. **Step 6** — classify vocabulary terms into `M`, `N`, `A`, `U`, and `O`;
7. **Step 7** — assign measures to statistical domains;
8. **Step 8** — identify semantic relationships between measures.

Run the pipeline in order:

```bash
python src/step1_extract_d_t.py
python src/step2_extract_s_t.py
python src/step3_build_geo_t_and_v_t.py
python src/step4_extract_titles.py
python src/step5_build_v_total.py
python src/step6_classify_category.py
python src/step7_domain_classification.py
python src/step8_semantic_relationships.py
```

Outputs are written to:

```text
results/step_outputs/2000/
```

or:

```text
results/step_outputs/7605/
```

depending on the selected dataset.

Step 7 and Step 8 outputs are grouped in dedicated `step7/` and `step8/` subdirectories.

## Evaluation

Manual evaluation scripts are provided for Steps 6, 7, and 8:

```text
evaluate/evaluate_step_6.py
evaluate/evaluate_step_7.py
evaluate/evaluate_step_8.py
```

The corresponding annotations and evaluation reports are stored under:

```text
results/evaluation/
```

The evaluation samples are tied to the datasets used to create the manual annotations:

- **Step 6:** 2,000-table subset;
- **Step 7:** 2,000-table subset;
- **Step 8:** full 7,605-table dataset.

Existing manually annotated evaluation files should normally be preserved rather than regenerated.

## Reference Resources

Reference files used by the pipeline are stored in:

```text
data/reference/
```

and include:

```text
NUTS2021-NUTS2024.xlsx
file-names_to_titles_eurostat_unfiltered.csv
domains_keywords.json
domain_descriptions.json
```

These resources are used for geographic information, title processing, and statistical-domain classification.

## Notes

Steps 7 and 8 use the `all-MiniLM-L6-v2` sentence-transformer model, which may be downloaded automatically on first use.