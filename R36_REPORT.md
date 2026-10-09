# R36：局部与PSD改善，完整响应和实测控制未通过

一次REAL-only条件去噪器训练完成2950更新，完整源驱动BASE经固定加噪和50步DDPM修正。它改善部分局部波形、PSD和条件峰宽，但完整形状、噪声RMS、源组、一个边界及REAL guide控制未共同通过。关闭这套固定条件扩散修正配方，不扫描强度、日程、反向方差或网络；项目仍active。

|同2000源父、同评价口径|BASE|R36|
|---|---:|---:|
|完整shape493 BA|.978|.958|
|纯u480 BA|.978|.954|
|完整电压480 BA|.982|.960|
|总体PSD W1|.02306546|.01870572|
|共同观察Q区1–7等权PSD W1|.03476580|.02901041|
|基线RMS W1，mV|.01935839|.03086203|
|基线ACF1 W1|.06423497|.03975358|
|actual-source purehad bin3谷比例，N213|5.6338%|14.5540%|
|actual-source purehad bin4谷比例，N181|.5525%|9.9448%|

full493对BASE的配对ΔBA为−.020，95%区间[−.039542,.000773]，未满足上界<0；纯u480的−.024、[−.045788,−.001972]通过。相对R35的full493又恶化.014，超过.01限制。完整BA仍高，不能以prompt .740、postpeak .828或baseline .664的局部改善替代完整响应判断。区间来自500次固定分类器留出区组重抽，不涵盖训练种子不确定性。

总体/共同Q区PSD W1分别下降18.90%/16.55%，原BASE有效密度区2–7全部保持有效。在这些共同区，低/高raw-width MAE由.010393/.013051到.005791/.007990，但峰位MAE由.005857/.002659到.010581/.006800，均变差。基线ACF1 W1改善38.11%，RMS W1却恶化59.42%。两处purehad谷比例分别增加8.92/9.39个百分点；全体source-Q/PSD秩关联.997418/.417549不能掩盖组内偏移。全部父、组成、顺序和权重保留，无非有限或ADC裁剪，417支持外源行回退BASE。

9个固定边界中8个通过。second_center_150的W1为.000776690V，超过.000703809V限；三个一阶边界均通过。同bin/条件/实际随机流的128个guide扰动全部改变输出，preADC配对RMS中位数.602099mV，大于1ADC LSB .061039mV。guide的Qfast逐值不变，Qtotal/PSD仅有已记录的float32微小数值差。此结果证明完整guide参与运算，不证明事件物理保真。

REAL guide控制使用2000条既有TRAIN_REFERENCE，其中1787支持内编辑、213支持外保留原实测ADC。其full493/u480/baseline BA为.598/.598/.594，原RAW REAL–REAL为.512/.510/.522；三项配对ΔBA的95%上界.129944/.141378/.125944均超过预定.10容差。完整电压BA由.546到.780，ΔBA区间[.178782,.285090]；其RMS W1约为RAW参照的5.04倍，这是描述结果，并非另设百分比等效门槛。控制包含见过的TRAIN样本，同一历史采集不能作为独立泛化证明。旧RAW参照没有Q/PSD二维预测，该配对项保留unknown，未补拟合。

六门依次失败/失败/失败/失败/失败/通过。独立CPU完整重放源2000和REAL控制2000，合计4000条，实际noise流、输入/支持、base/edit精确分量及pre/postADC全部逐值一致。最终物理signal/electronic拆分unknown。原评价缺失旧2D参照时在任何HGB/EM前失败，修复后只拟合原登记20个新HGB与16次EM；没有重复生成或旧参照拟合。

累计科学859.986274/3000秒，包含失败和绘图，元数据预检及封存时间分列；峰RSS 1,950,699,520字节、GPU allocated 353,069,056字节、CPU4。一次远端RSS查询缺测原因unknown。10张最终图实际查看，2张布局修复前图另保留，根实际查看6张关键图。实际封存native0，309文件/183,809,685字节。

下一轮R37检验“冻结源神经G + TRAIN完整协方差线性校准”。使用原94400行流拟合8个480维高斯仿射映射，无新增神经更新；唯一正定化方案、不用DEV拟合、不扫描参数。峰位偏差预先加入共同门槛。完整源仍经G和矩阵进入输出，不能退回目标随机自生成。

[完整执行报告](work/R36_CONDITIONAL_DIFFUSION_EDIT_001/REPORT.md) · [完整形状与实测控制](work/R36_CONDITIONAL_DIFFUSION_EDIT_001/results/figures/c2st.png) · [电荷–PSD](work/R36_CONDITIONAL_DIFFUSION_EDIT_001/results/figures/charge_psd.png) · [条件峰位/峰宽](work/R36_CONDITIONAL_DIFFUSION_EDIT_001/results/figures/energy_peak_width.png) · [guide控制与输入作用](work/R36_CONDITIONAL_DIFFUSION_EDIT_001/results/figures/real_control_and_guide.png) · [复现入口](work/R36_CONDITIONAL_DIFFUSION_EDIT_001/REPRODUCTION.md) · [R37协议](tasks/R37_FULL_COVARIANCE_CALIBRATION_001.md)。

实际私有Git和D副本完成情况见R36_PUBLICATION_RECEIPT.json与R36_BACKUP_RECEIPT.json。未访问新REAL、CAL、EVAL或G4，完整项目验收尚未通过。
