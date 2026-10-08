from __future__ import annotations
import pathlib, json, hashlib, datetime, os, ast, struct
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent.parent
def write(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def bound(path,n,sha):
    p=pathlib.Path(path)
    if not p.is_absolute():p=BASE/p
    if p.stat().st_size!=n:raise RuntimeError('size binding '+str(p))
    b=p.read_bytes()
    if hashlib.sha256(b).hexdigest()!=sha:raise RuntimeError('SHA binding '+str(p))
    return b
def main():
    ROOT.mkdir(exist_ok=True)
    freeze_bytes=bound('FROZEN_METHOD.json',10513,'b4e1f86807dba40f6004bda48a40d0644fe5039535a3c3eb2e402fc213065186')
    frozen=json.loads(freeze_bytes);bindings=[]
    for s in frozen['model_and_generation_bindings']+frozen['definitions_and_tasks']+frozen['historical_controls']:
        bound(s['path'],s['bytes'],s['sha256']);bindings.append(s)
    j=BASE/'work/P1J_CURRENT_ROLE_INDEX_001/run_001'
    role_index=json.loads((BASE/'CURRENT_ROLE_INDEX.json').read_text())
    manifest_bytes=bound(j/'ARTIFACT_MANIFEST.json', (j/'ARTIFACT_MANIFEST.json').stat().st_size,role_index['payload_manifest_sha256'])
    manifest=json.loads(manifest_bytes);specs={s['name']:s for s in manifest['members']}
    data={n:bound(j/n,specs[n]['bytes'],specs[n]['sha256']) for n in ['role.bin','source_id_sha256.bin','group.bin']}
    indices=[i for i,r in enumerate(data['role.bin']) if r==3]
    if len(indices)!=7600 or len(set(i//50 for i in indices))!=152:raise RuntimeError('fixed EVAL selection')
    if any(struct.unpack_from('<H',data['group.bin'],i*2)[0]!=i//50 for i in indices):raise RuntimeError('artificial group identity')
    digest=b''.join(data['source_id_sha256.bin'][i*32:(i+1)*32] for i in indices)
    (ROOT/'eval_original_index.bin').write_bytes(struct.pack('<7600I',*indices))
    (ROOT/'eval_source_id_sha256.bin').write_bytes(digest)
    write('EVAL_IDENTITIES.json',{'original_index':indices,'source_id':['data:source:'+digest[i*32:(i+1)*32].hex() for i in range(7600)],'role':3,'artificial_groups':152,'source_waveform_values_decoded':0})
    old=BASE/'work/R3B_ENERGY_CONDITIONAL_WIDTH_001'
    for n in ['common_math.py','evaluation.py','energy_width.py']:
        source=(old/n).read_text(encoding='utf-8')
        if n=='evaluation.py':
            source=source.replace('np.zeros(2000,dtype=',"np.zeros(len(d['V']),dtype=").replace('np.full((2000,2)',"np.full((len(d['V']),2)")
            source=source.replace("'source_n':2000 if r is None", "'source_n':len(d['V']) if r is None")
        if n=='energy_width.py':source=source.replace('import torch\n','')
        (ROOT/n).write_text(source,encoding='utf-8')
    (ROOT/'interface_core.py').write_text((BASE/'work/P1K_REAL_REFERENCE_AND_REPEATABILITY_001/interface_core.py').read_text().replace('if position<2000:', 'if old_real is not None and position<2000:'),encoding='utf-8')
    oldplan=json.loads((BASE/'work/P1K_REAL_REFERENCE_AND_REPEATABILITY_001/PROTOCOL.json').read_text())
    source=oldplan['remote_source'];source['io_caps']['hash']=0
    source['io_caps']['metadata']=262144
    source['io_caps']['source_total']=source['io_caps']['compressed']+262144
    source['memory_limit_bytes']=1073741824
    config=json.loads((old/'config.json').read_text())['evaluation']
    config.update({'stage':'R4_FROZEN_HELDOUT_RESPONSE_001','start_UTC':'2026-10-08T11:22:42.929665+00:00',
                  'deadline_UTC':'2026-10-08T12:52:42.929665+00:00','numeric_wall_limit_s':1800,'SSH':1,
                  'EVAL_read':7600,'response_training':0,'remote_source':source,
                  'remote_python':'/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python',
                  'remote_numpy_version':'1.26.4','main_REAL_rule':'SHA256 full digest R4-main-real-v1|source_id ascending,QC only,first2000',
                  'RR_group_rule':'SHA256 full digest R4-RR-group-v1|repeat|group ascending,first76 vs last76',
                  'RR_event_rule':'SHA256 full digest R4-RR-event-v1|repeat|source_id ascending,first2000 each half',
                  'sliced_W1':{'angles':128,'range':'[0,pi)','standardization':'TRAIN-fit2592 mean and population SD of logQ and PSD only'},
                  'all_candidates':['SIM','GEN_W0','R2','R3','R3B'],'no_EVAL_tuning':True,
                  'source_sha':'inherited full-container SHA; current read validates bound central/local headers and complete selected-member CRC/EOF; no whole-container rehash',
                  'metadata_unselected_arrays_decode':0})
    write('config.json',config)
    execute=[]
    for p in sorted(ROOT.glob('*.py')):
        b=p.read_bytes();ast.parse(b);execute.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    write('READINESS.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),
                          'FROZEN_METHOD_SHA256':hashlib.sha256(freeze_bytes).hexdigest(),'frozen_bindings_checked':bindings,
                          'execution_sources':execute,'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),'tasks':frozen['definitions_and_tasks'][1:4],
                          'EVAL_scientific_values_before_this_record':0,'new_response_fits':0,
                          'selections':{n:{'bytes':(ROOT/n).stat().st_size,'sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest()} for n in ['eval_original_index.bin','eval_source_id_sha256.bin']},
                          'historical_exposure':'same single acquisition; prior event-level exposure unknown; current response fit isolation only',
                          'future_R5_R6_are_frozen':True})
    print(json.dumps({'kind':'R4_READINESS_written','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_files':len(bindings),'EVAL_values':0}),flush=True)
if __name__=='__main__':main()
