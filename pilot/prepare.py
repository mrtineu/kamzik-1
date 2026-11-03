"""Prepare pinned models and configured public datasets; retain legacy GSM8K runs."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for k,v in {'HF_HOME':ROOT/'.cache/huggingface','XDG_CACHE_HOME':ROOT/'.cache','TORCH_HOME':ROOT/'.cache/torch','TRITON_CACHE_DIR':ROOT/'.cache/triton','TMPDIR':ROOT/'.cache/tmp'}.items():
    Path(v).mkdir(parents=True,exist_ok=True)
    os.environ[k]=str(v)
os.environ['HF_HUB_DISABLE_XET']='1'
import argparse,hashlib,json,urllib.request
from huggingface_hub import HfApi,snapshot_download,hf_hub_download
from common import write_json
from task_data import DATASETS,dataset_specs


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare_gsm8k(manifest):
    prior=manifest.get('datasets',{})
    ds_sha=prior.get('train',{}).get('url','').split('/')[5] if prior.get('train') else json.load(urllib.request.urlopen('https://api.github.com/repos/openai/grade-school-math/commits/master',timeout=60))['sha']
    for split in ['train','test']:
        url=f'https://raw.githubusercontent.com/openai/grade-school-math/{ds_sha}/grade_school_math/data/{split}.jsonl'
        local=ROOT/f'data/gsm8k_{split}.jsonl'
        raw=local.read_bytes() if local.exists() else urllib.request.urlopen(url,timeout=60).read()
        sha=hashlib.sha256(raw).hexdigest()
        if split in prior and sha!=prior[split]['sha256']:
            raise RuntimeError('Dataset hash mismatch: '+split)
        local.parent.mkdir(parents=True,exist_ok=True)
        if not local.exists():local.write_bytes(raw)
        manifest['datasets'][split]={'url':url,'sha256':sha}
    write_json(ROOT/'manifest.json',manifest)


def prepare_hf_dataset(name,manifest,api):
    spec=DATASETS[name]
    prior=manifest['datasets'].get(name,{})
    if prior and prior.get('repo')!=spec['repo']:
        raise RuntimeError('Dataset repository changed: '+name)
    revision=prior.get('revision') or api.dataset_info(spec['repo']).sha
    entry={**prior,'repo':spec['repo'],'config':spec['config'],'revision':revision,'splits':dict(prior.get('splits',{}))}
    # Save the revision before downloading, so an interrupted preparation stays pinned.
    manifest['datasets'][name]=entry
    write_json(ROOT/'manifest.json',manifest)
    files=None
    for split in spec['splits']:
        target=ROOT/'data'/f'{name}_{split}.jsonl'
        saved=entry['splits'].get(split)
        if target.exists() and saved:
            if digest(target)!=saved['sha256']:
                raise RuntimeError(f'Dataset hash mismatch: {name}/{split}')
            print(f'Verified {name}/{split}: {saved["rows"]} rows',flush=True)
            continue
        if target.exists():
            raise RuntimeError(f'Unverified dataset file already exists: {target}')
        import pyarrow.parquet as pq
        if files is None:files=api.list_repo_files(spec['repo'],repo_type='dataset',revision=revision)
        prefix=spec['config'] if name=='arc_easy' else 'data'
        shards=sorted(f for f in files if f.startswith(f'{prefix}/{split}-') and f.endswith('.parquet'))
        if not shards:raise RuntimeError(f'No Parquet files for {name}/{split} at {revision}')
        sources=[];count=0
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.jsonl.tmp')
        with temporary.open('w') as output:
            for filename in shards:
                print(f'Downloading {name}: {filename} at {revision}',flush=True)
                source=Path(hf_hub_download(spec['repo'],filename,repo_type='dataset',revision=revision))
                sources.append({'filename':filename,'sha256':digest(source)})
                for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        output.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
        normalized_hash=digest(temporary)
        if saved and (normalized_hash!=saved['sha256'] or sources!=saved['sources']):
            raise RuntimeError(f'Redownloaded dataset differs from pinned hashes: {name}/{split}')
        temporary.replace(target)
        entry['splits'][split]={'file':str(target.relative_to(ROOT)),'sha256':digest(target),'rows':count,'sources':sources}
        write_json(ROOT/'manifest.json',manifest)
        print(f'Prepared {name}/{split}: {count} rows',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=os.environ.get('QRL_CONFIG','config.json'))
    parser.add_argument('--datasets-only',action='store_true',help='Skip model downloads')
    args=parser.parse_args()
    c=json.loads((ROOT/args.config).read_text())
    specs=dataset_specs(c)
    manifest=json.loads((ROOT/'manifest.json').read_text()) if (ROOT/'manifest.json').exists() else {'models':{},'datasets':{}}
    api=HfApi()
    if not args.datasets_only:
        sha=manifest['models'].get('student',{}).get('revision') or api.model_info(c['student']).sha
        manifest['models']['student']={'repo':c['student'],'revision':sha}
        write_json(ROOT/'manifest.json',manifest)
        print('Downloading student',sha,flush=True)
        snapshot_download(c['student'],revision=sha,local_dir=ROOT/'models/student',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
        sha=manifest['models'].get('teacher',{}).get('revision') or api.model_info(c['teacher_repo']).sha
        manifest['models']['teacher']={'repo':c['teacher_repo'],'revision':sha,'filename':c['teacher_file']}
        write_json(ROOT/'manifest.json',manifest)
        print('Downloading quantized teacher',sha,flush=True)
        hf_hub_download(c['teacher_repo'],c['teacher_file'],revision=sha,local_dir=ROOT/'models/teacher')
    if not specs or any(s['name']=='gsm8k' for s in specs):prepare_gsm8k(manifest)
    for spec in specs:
        if spec['name']!='gsm8k':prepare_hf_dataset(spec['name'],manifest,api)
    if specs:
        from task_data import load_pools
        for spec in specs:
            counts={split:len(rows) for split,rows in load_pools(ROOT,spec).items()}
            print(f'{spec["name"]}: eligible unique rows {counts}',flush=True)
    print('Preparation complete',flush=True)

if __name__=='__main__':main()
