#!/bin/bash
# Run the whole test suite in the environment each test needs and write results/tables/test_report.csv (for the presentation).
#   training environment: INSPIRE equivalence tests (torch), GPU training test, reference tests of the Leiden steps
#   analysis environment: synthetic tests, reference tests of the deterministic analyses
source "$(dirname "$0")/../common.sh"
$PYT -m pytest -q -p no:cacheprovider -W ignore tests/test_inspire.py tests/test_training_reference.py --junitxml=logs/tests_train.xml "$@"
$PY -m pytest -q -p no:cacheprovider -W ignore tests/test_areal.py tests/test_metrics.py tests/test_columns_fgw.py tests/test_reference.py --junitxml=logs/tests_analysis.xml "$@"
$PY - <<'EOF'
import xml.etree.ElementTree as ET, pandas as pd, os
rows = []
for f in ('logs/tests_train.xml', 'logs/tests_analysis.xml'):
    if not os.path.exists(f): continue
    for tc in ET.parse(f).getroot().iter('testcase'):
        out = 'failed' if tc.find('failure') is not None or tc.find('error') is not None else 'skipped' if tc.find('skipped') is not None else 'passed'
        rows.append(dict(file=tc.get('classname', '').split('.')[-1], test=tc.get('name'), outcome=out, seconds=float(tc.get('time', 0))))
os.makedirs('results/tables', exist_ok=True); D = pd.DataFrame(rows); D.to_csv('results/tables/test_report.csv', index=False)
print(D.groupby('file').outcome.value_counts().unstack(fill_value=0).to_string())
EOF
