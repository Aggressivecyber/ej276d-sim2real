import pathlib,json,subprocess,os,datetime
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());base=cfg['remote_root'].rsplit('/',1)[0];records=[]
for folder,remote in [('attempt_01',base),('attempt_02',base+'/attempt_02')]:
    argv=['scp','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',cfg['execution']['location']+':'+remote+'/UPLOAD.tar',str(ROOT/folder/'EXECUTED_SOURCE_UPLOAD.tar')]
    p=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));out,err=p.communicate(timeout=30)
    records.append({'local_snapshot':folder+'/EXECUTED_SOURCE_UPLOAD.tar','remote':remote+'/UPLOAD.tar','PID':p.pid,'exit_code':p.returncode,'stdout':out.decode(errors='replace'),'stderr':err.decode(errors='replace'),'source_only_original_upload_retained':True,'migration_equality_test':False})
    if p.returncode:break
(ROOT/'FAILURE_SOURCE_RETENTION_RECEIPT.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'no_scientific_payloads_or_new_job':True,'patched_loose_source_snapshots_not_authoritative_for_old_failures':'original executed tar archives above are authoritative'},indent=2)+'\n');print(json.dumps(records),flush=True)
raise SystemExit(0 if len(records)==2 and all(r['exit_code']==0 for r in records)else 1)
