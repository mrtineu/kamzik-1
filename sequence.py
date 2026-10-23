"""Run the quick pilot, then the larger experiment without a live terminal."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from common import write_json,read_jsonl


def main():
    env=os.environ.copy()
    env.update(HF_HOME=str(ROOT/'.cache/huggingface'),XDG_CACHE_HOME=str(ROOT/'.cache'),
               TORCH_HOME=str(ROOT/'.cache/torch'),TRITON_CACHE_DIR=str(ROOT/'.cache/triton'),
               TMPDIR=str(ROOT/'.cache/tmp'),HIP_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
    for name,config in [('quick','config_quick.json'),('pilot','config.json')]:
        run=ROOT/'runs'/name;run.mkdir(parents=True,exist_ok=True)
        if (run/'complete.json').exists():continue
        run_env={**env,'QRL_RUN':name,'QRL_CONFIG':config}
        with (run/'console.log').open('a',buffering=1) as logfile:
            process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'experiment.py')],cwd=ROOT,env=run_env,
                                     stdout=logfile,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write_json(ROOT/'runs/active.json',{'supervisor_pid':os.getpid(),'process_pid':process.pid,'run':name,'state':'running'})
            try:code=process.wait()
            except KeyboardInterrupt:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=20)
                write_json(ROOT/'runs/active.json',{'supervisor_pid':os.getpid(),'run':name,'state':'stopped'})
                return 130
        if code:
            write_json(ROOT/'runs/active.json',{'run':name,'state':'failed','exit_code':code})
            return code
        if name=='quick':
            # Exactly the same training prefix, batch grouping, prompts, and RNG seeds.
            # Reuse generated training data, never the quick-run trained weights.
            for filename in ['coaching.jsonl','candidates_0.jsonl','candidates_1.jsonl']:
                old=read_jsonl(ROOT/'runs/pilot'/filename)
                ids={r['id'] for r in old}
                new=[r for r in read_jsonl(run/filename) if r['id'] not in ids]
                if new:
                    with (ROOT/'runs/pilot'/filename).open('a') as f:
                        for r in new:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(ROOT/'runs/active.json',{'state':'complete','completed_at':time.strftime('%Y-%m-%d %H:%M:%S')})
    return 0

if __name__=='__main__':
    lock=(ROOT/'runs/.sequence.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A sequence is already running.')
    raise SystemExit(main())
