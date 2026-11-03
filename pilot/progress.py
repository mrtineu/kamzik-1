"""Show process state, generation counts, and any completed comparison."""
import argparse
import json
from pathlib import Path

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',action='append',help='Run to inspect; repeat for multiple runs (default: all experiments)')
args=parser.parse_args()
if (root/'runs/active.json').exists():
    print((root/'runs/active.json').read_text())
names=args.run or sorted(p.name for p in (root/'runs').iterdir() if p.is_dir() and (p/'config.json').exists())
for name in names:
    run=root/'runs'/name
    print(name.upper())
    if (run/'status.json').exists():print((run/'status.json').read_text())
    for file in sorted(run.glob('*.jsonl')):
        if file.name.startswith('rejected_'):continue
        rows=[json.loads(x) for x in file.read_text().splitlines() if x.strip()]
        print(f'  {file.stem}: {len(rows)}'+(f', correct {sum(r.get("correct",False) for r in rows)}' if rows and 'correct' in rows[0] else ''))
    if (run/'complete.json').exists():
        print((run/'RESULTS.md').read_text())
