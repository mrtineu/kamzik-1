"""Stop this project's running sequence and allow normal checkpoint cleanup."""
import json,os,signal
from pathlib import Path
root=Path(__file__).resolve().parent
path=root/'runs/active.json'
if not path.exists():raise SystemExit('No recorded sequence.')
state=json.loads(path.read_text())
if state.get('state')!='running':raise SystemExit('No active sequence: '+state.get('state','unknown'))
pid=state['supervisor_pid']
try:
 command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 if str(root/'sequence.py') not in command:raise SystemExit('PID does not belong to this project; no signal sent.')
 os.kill(pid,signal.SIGINT)
 print('Stop requested. Completed generation outputs remain resumable.')
except FileNotFoundError:print('Sequence is no longer running.')
