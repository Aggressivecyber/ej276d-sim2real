# R72 公共完整波形响应候选

实现遵从根最终固定的唯一架构：`482→128→128→480`、两次 SiLU、float32 网络，末层零初始化；未运行32宽草案。R71唯一最终Q头由根加载并冻结为float64，始终输入原始480点。其余 lifting、连续模板、完整噪声、原64次Box使用float64，pre-ADC为float32，再用原canonical ADC；训练可显式启用原STE。该模块不读取科学文件、不拟合统计、没有训练循环、没有域/父ID/h/标签或epsilon网络输入。

## 固定接口

```python
shape = ShapeResidualNetwork(feature_center482, feature_scale482, hidden=128)
prepared = prepare_static(head, x, mu, scale, edges, p480, tau2)
# prepared['features482'] 是未标准化输入，可由根仅用合法REAL TRAIN拟合统计。
part = slice_prepared(prepared, start, stop)
pre32, canonical, raw64, meta = forward_prepared(shape, part, epsilon,
                                                straight_through=True)
pre32, canonical, raw64, meta = forward_raw(shape, head, x, epsilon,
                                          mu, scale, edges, p480, tau2,
                                          straight_through=False)
```

网络构造接受根已拟合的482维均值与总体std（已floor到1e-6），自身不拟合或暗加floor；缓冲区float64、参数float32，移动设备用`.to(device)`，不能用`.float()`把标准化统计一并改成float32。准备函数要求Q头确实冻结，返回detached公共原输入前置。扰动、新输入或冻结Q/统计变化时重新调用公开入口，不可依据父ID复用。`slice_prepared`只切显式ROW_KEYS，共享9边界和241谱不切。

输入为 `[(lifted-mean100(lifted))/Q*,log(Q0),log(Q*)]`，最后两项使用原窗积分数值单位的自然对数。所有480点进入MLP，无跨样本统计层。R71 `scale_condition` 是旧 `scale_unit_q` 的连续插值，单位电荷尺度直接作电压残差会少Q*；根已明确修正为 `d_raw=Q* scale_condition f`、`d=d_raw-m q(d_raw)`。原32宽草案到128的选择为预登记容量决定，没有根据科学结果挑宽度。

返回前置包含 features482、lifted、mu_condition、scale_condition、residual_scale_V、Q0/Qstar、原/目标raw/clipped bins、连续Q条件/固定u斜率、lifted前100点unbiased方差、R71噪声幅度平方/幅度/kappa、共享edges_logQ/noise_sqrt_p480。完整epsilon由调用者显式传入，转换float64后按原rFFT/irFFT ortho着色、同一mu零Q投影，再乘冻结R71 completion幅度；此幅度依据lifted，不依据本轮形状残差重拟合。

`raw=lifted+noise+d`，Box目标为独立预测Q*；不裁行、不按支持区过滤、不把Q*重定义为波形积分。meta保存原/Q*/pre/ADC电荷与四阶段raw/clipped区迁移、所有支持外标记、两种Q*相对误差、残差Q/RMS、projection活动与RMS、噪声条件和Q比值。仅 `range_projection_fraction` 是零维批汇总；跨chunk应聚合逐行 `range_projection_active`，其余meta为逐行。

## 独立参考与人工入口

`waveform_numpy_reference.reference_numpy(q_state,shape_state,x,epsilon,mu,scale,edges,p480,tau2)` 返回同四元组，纯NumPy、无Torch和文件读写。shape float32权重提升到float64后独立算MLP SiLU，公共Q头、模板插值、稠密噪声协方差、64次Box及原float32 legacy→canonical ADC独立实现；这是数学数值参考，不是第二个模型。零shape另外对照旧封存R71 NumPy参考。

`check_waveform_response.run_artificial(device='cpu')` 可由根在注册后显式使用CUDA；导入不创建模型/读文件，检查内恢复CPU和指定CUDA RNG。`make_fixture(device)` 返回8人工父的x/epsilon/mu/scale/edges/p480/tau2，包含区外父但未覆盖完整分区人口，不能冒充完整objective fixture；根另负责全部loss与两遍VJP人工准入。本地native launcher先加载既有local_resource_guard，再导入Torch，所有临时目录在本候选E路径。

第一次native检查在严格按字节的row permutation断言失败，失败回执完整保留为ATTEMPT_01_*。诊断确认前置features/Q*按字节相同；float32 MLP行位置舍入导致神经输出最高差9.31e-10，raw最高差8.52e-10V，pre32最高差1.49e-8V、最高RMS2.25e-5 LSB。该人工场景canonical按字节相同，ADC整数码差0。不能据无跨样本层宣称普遍换行逐位等变；需分别记录数学/精度内等变与字节结论，不更改科学数值或重放阈值。

最终人工结果、实际native退出码/资源及源码哈希以同目录回执为准。人工检查不能证明科学收益、数小时总成本、同处理REAL分布保持、缺口有效性或最终A–F。

最终本机CPU native共38项检查通过，实际退出码0，PID67200，affinity mask2/单逻辑核、Torch和interop各1线程、既有1GiB硬RSS上限，峰值RSS230,481,920字节，约219.8MiB，guard内墙钟4.464328秒。未调用CUDA、未读科学数组或权重、未运行科学训练/探针；实际人工张量和自动微分仅核工程实现。独立非零残差参考最高pre-ADC RMS为5.5791087071632055e-6 LSB，Q*相对误差最高float64为3.6725442024451473e-16、float32为5.33230354617444e-9。

根定位末位GEMM舍入后确认，将本fixture的换行几何工程断言修正为atol/rtol各2×float32 epsilon（2.384185791015625e-7），另核同fixture实际ADC码相等；这不是科学数值门的更改。最终回执明确保存pre32/raw64换行不按字节相同，canonical在此fixture按字节相同。首轮失败回执、逐项差异诊断均与成功回执一并封存，不能据这8行fixture外推普遍ADC码或byte permutation等变。

## 根训练器静态接口核对

已只读根的train_full.py/postcheck.py。prepare/slice/forward与独立reference参数顺序、网络统计字段、元数据字段匹配；postcheck跳过零维汇总并以活动向量统计投影率。完整两遍使用同一个epsilon、同chunk范围，缓存根只用于固定公共原输入。逐项`autograd.grad`默认释放图，根必须确保objective.iter_terms各项重建自己的非叶图或显式保留共享图；完整人工objective+VJP检查应覆盖，不在此替代。未改根训练器、未执行该模块。

当前数值检查的Python bool在CUDA有同步，完整每步计时应包含它。可后续把等价错误标志在GPU聚合再一次读回，保持原错误/边界语义；本候选未实施此优化，也未改变Box内部检查或迭代。独立NumPy参考在ADC离散边界的差异必须按原门评估。最终重放保持同样chunk形状可避免正常GEMM分组差异；根的严格重放门仍保留。

根在科学执行前将训练重放chunk从1024草稿固定为27640完整人口，不改变本模块。静态未见由此引入的跨行状态：MLP无BatchNorm/跨行统计，Box逐行并为自定义autograd，64次二分不保留64层自动微分图，反向只保存free掩码及电荷算子。一份27640×480 float64张量约106.1MB，参数140,256个约0.56MB；完整前置、多端点叶、float64梯度累加和Box临时张量会叠加，不能据此宣称总GPU必低于10GiB。根需用已登记资源上限测完整更新，不能以OOM为由自动删除人口。generation256与fullN的通常GEMM舍入差由独立数学参考门检查；固定生成chunk的保存/重放逐位检查仍是另一个约束。本机结果不是fullN CUDA内存、完整objective/VJP或总训练成本证明。
