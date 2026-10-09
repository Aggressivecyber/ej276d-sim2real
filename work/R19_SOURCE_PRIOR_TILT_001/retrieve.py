import pathlib,json,subprocess,tarfile,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());remote=cfg['execution']['remote_root'];host=cfg['execution']['location'];opts=['-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8'];start=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=subprocess.run(['scp',*opts,host+':'+remote+'/RETURN.tar',str(ROOT/'REMOTE_RETURN.tar')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));receipt={'PID':os.getpid(),'start_UTC':start,'exit_code':p.returncode,'stdout':p.stdout.decode(errors='replace'),'stderr':p.stderr.decode(errors='replace'),'migration_content_validation':False,'normal_transfer_only':True}
(ROOT/'RETURN_TRANSFER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
if p.returncode:raise SystemExit(p.returncode)
with tarfile.open(ROOT/'REMOTE_RETURN.tar') as t:
    members=t.getmembers();total=sum(m.size for m in members)
    if total+(ROOT/'REMOTE_RETURN.tar').stat().st_size>cfg['output_limit_bytes']:raise RuntimeError('return artifact cap')
    for m in members:
        if not m.isfile() or '..' in pathlib.PurePosixPath(m.name).parts or pathlib.PurePosixPath(m.name).is_absolute():raise RuntimeError('unsafe return member')
        if (ROOT/m.name).exists():raise RuntimeError('return would overwrite existing local file '+m.name)
    t.extractall(ROOT,filter='data')
receipt['return_tar_bytes']=(ROOT/'REMOTE_RETURN.tar').stat().st_size;receipt['returned_file_count']=len(members);receipt['stop_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();(ROOT/'RETURN_TRANSFER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
