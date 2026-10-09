# R36 复现与保存入口

本目录已执行证据，不是再次运行许可。训练仅一次，固定最终权重；生成/评价/完整主回放均在本机CPU4执行。旧阶段只读。

训练服务器目录：`/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R36_REMOTE_20261009_194711_UTC`；解释器：`/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`。正常部署由deploy_training.py、launch_remote.py和linux_supervise.py/train_server.py完成；训练actual50548/super50547，原生exit0见TRAINING_NUMERIC_RECEIPT.json和remote_receipts/train_server.py.NATIVE_RECEIPT.json。服务器既有Python3.12.3/NumPy1.26.4/torch2.12.0+cu130、RTX5070；训练输入复用远端R33的TRAIN target/原实际流和仿射，不需要新G4、原长采集或EVAL。

本机解释器：`C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，Python3.13.3/NumPy2.2.6/torch2.10.0+cpu/sklearn1.7.2。实际入口形式：`python -B supervise.py <entry.py> <CPU_python>`。已执行generate_all.py（PID30272/native0）、evaluate_saved_repair.py（28588/native0）、replay_all.py（53452/native0）、summarize_and_plot.py（60552/native0）；原evaluate_saved.py首次失败留档。supervise.py使用entry.used防重复提交、导入前真实PID/nonce握手和RSS守护；复现实验需复制到新的E/D目录，并明确获得新预算，不能删掉本目录used记录重跑。

部署推理最小依赖为diffusion_core.py、flow_runtime.py、model.py/legacy_model.py、flow_core.py，冻结BASE G（`E:\GAN\research\sim2real_two_papers_20261008_0305\work\R32_CONDITIONAL_MEAN_TRANSLATION_001\SERVER_FRESH_CONTROL\FINAL_TRANSLATOR.pt`）、CONDITIONAL_DIFFUSION_EDIT/FINAL_DENOISER.pt、DIFFUSION_CONSTANTS.npz及保存record seeds。`Pipeline`先对完整source执行BASE，`set_seeds`后`infer`执行固定50步编辑，最后model.adc一次。它不加载REAL训练库。condition来自实际guide；支持来自原未量化source-Q。不能从扰动guide重算固定探针condition。

已有完整源输入：ACTUAL_GENERATION_INPUTS.npz（source_waveforms_V/source_ids及原z/white/style）、DEV_SOURCE_METADATA.npz；REAL control仅使用REAL_DEV_AND_REFERENCE_PRIVATE.npz既存reference_V/IDs/groups。JOINT8_INPUTS_PRIVATE.npz、PROBE_INPUTS_CORRECTED.npz及ACTUAL_DIFFUSION_SEED_MAPS_PRIVATE.npz保存辅助/种子关联。条件常数/日程在CONDITION_CONSTANTS.json和DIFFUSION_CONSTANTS.npz；核心源码冻结哈希在PRETRAIN_FREEZE/PRE_GENERATION_SOURCE_FREEZE/SOURCE_BINDINGS。

主结果：CONDITIONAL_DIFFUSION_EDIT/generated.npz与REAL_GUIDE_EDIT/generated.npz保留精确pre_ADC_V、waveforms_V、base_pre_ADC_V、flow_increment_f64_V、support、conditions、record seed、实际noise hash和身份。flow命名只是兼容编辑增量，不代表使用旧CFM。额外guide原/扰动/输出与固定c在results/GUIDE_PERTURBATION_PRIVATE.npz。原full480/source-probe/joint实际随机序列保持完整，不用ID重新抽噪声。

REPLAY_RECEIPT.json由独立CPU进程实际完成4000条全链逐字节比较，包括精确噪声哈希/条件/支持/父ID和BASE/edit分量，全部通过；未重拟合任何模型。源主preADC数组SHA b82aa8763568287917fe0cf7101471159231d681c0217b95db539bfa41c0d073、ADC数组SHA e11f1dd8072342aebb1f6eb5de4b33bf41bc509dd2cf35b9c0bd712cd6587b04。REAL控制对应数组SHA见回放回执。数组SHA是已执行回放证据，不是跨端迁移测试。

评价复用原600/250分组、旧BASE/R35/RAW REAL–REAL的预测及既有TRAIN/Q-bin/密度定义；新20HGB与16EM均已保存到results。完整metrics/预测/身份/噪声为本地私有结果，METRICS_COMPACT仅聚合。没有EVAL、新REAL或CAL波形解码、原长波形读取、安装或新增G4。

恢复到新目录时按config中明确路径恢复这些既有输入和权重、保持core/常数/实际随机映射/CPU线程和版本；更改项目根只调整路径映射，不更改科学配方。所有缓存TEMP/TMP/TORCH_HOME/CUDA_CACHE_PATH/MPLCONFIGDIR/JOBLIB_TEMP_FOLDER设E/D。不同平台训练不声称CPU/CUDA逐位一致；本次完整推理与独立推理均是已记录本机环境。

这是同采集、已历史暴露的开发诊断，不是新盲测。固定配方联合门槛失败；可复现性不等于响应物理真实性。
