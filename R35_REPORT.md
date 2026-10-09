# R35：总体PSD改善，源组与噪声未保住，配对调整分支关闭

唯一PSD同序耦合CFM完成2950更新。它相对冻结BASE改善完整形状分类和总体PSD距离，但纯had源组谷占比、基线噪声和原有效密度支持失败。关闭当前冻结BASE加480维MLP-CFM的耦合、代价及其他小变体；不把单项PSD收益视为完整响应验收。项目仍active。

|同2000源父、同评价口径|BASE|R35|
|---|---:|---:|
|完整shape493 BA|.978|.944|
|纯u480 BA|.978|.952|
|完整电压480 BA|.982|.966|
|总体PSD W1|.02306546|.01530813|
|共同观察Q区1–7等权PSD W1|.03476580|.03712954|
|基线RMS W1，mV|.01935839|.02239106|
|基线ACF1 W1|.06423497|.08363765|
|actual-source purehad bin3谷比例，N213|5.6338%|11.7371%|
|actual-source purehad bin4谷比例，N181|.5525%|10.4972%|

完整shape493/纯u480对BASE配对ΔBA为−.034/−.026，95%区间分别[−.058929,−.008686]/[−.052292,−.001737]；相对原Independent为−.004/0，区间均跨0，未证明胜过该流。绝对BA仍高，电压480对原Independent的.954也变差。500次留出区组重抽条件于固定分类器，不覆盖训练种子或独立采集不确定性。

总体PSD W1下降33.63%，条件Q区W1上升6.80%。共同有效密度区2–6的低/高raw-width MAE由.011641/.015332到.011618/.007952；与此同时，低/高峰位MAE由.004416/.002450到.022100/.014941，明显变差。原BASE有效bin7在本轮500步EM未收敛，不能只报告缩小交集上的宽度收益；bin8成分支持不足也保留。

全体source-Q/PSD Spearman为.998535/.420039，后者恢复到BASE附近，却未保住源组内分布：两个purehad区谷比例分别增加6.10/9.94个百分点，超过2个百分点限制。基线RMS/ACF1 W1分别恶化15.67%/30.21%，均超过10%限制。全部源ID、组成、顺序和权重保留，无新增无效或ADC裁剪；原417支持外行回退BASE，未删除。谷占比和源敏感性是保真代理，不是逐事件物理真值。

9个固定边界差均通过，未施加分段投影或平滑；32vs64 Heun最大preADC差1.73524e−5V低于1/16383V，发布仍32步。独立CPU实际重放完整source→冻结G→流→一次ADC共2000条，原输入/支持与精确分量和pre/postADC逐值一致。实际完整源来自既有EE/had总沉积派生，微观沉积时序未恢复；最终物理signal/electronic拆分unknown。

四门依次为通过/失败/失败/通过。原评价在已保存全部10HGB、8EM和生成结果后因旧域名聚合失败；恢复仅续算聚合，未重复拟合或生成。累计科学82.380894秒，含元数据预检保守83.977473/1800秒；真实峰RSS 2,016,718,848字节、GPU allocated 148,003,840字节，CPU4。远端一次RSS缺测保留unknown。13张最终图实际查看，CDF布局修复前图另保留，根实际查看6张关键图。实际封存native0，253文件/142,172,481字节。

下一轮R36已登记一个实质不同的问题：学习REAL域内条件去噪，从完整源驱动BASE的有限加噪guide进行固定50步修正，用电荷/PSD条件、实测guide控制及同条件同噪声的完整guide扰动共同检验。唯一2950更新/900秒训练、累计科学3000秒，不扫描加噪强度或网络；完整源仍须影响输出，不能退回目标随机自生成。

[完整执行报告](work/R35_PSD_ORDERED_FLOW_001/REPORT.md) · [电荷–PSD图](work/R35_PSD_ORDERED_FLOW_001/results/figures/charge_psd.png) · [条件峰位与峰宽](work/R35_PSD_ORDERED_FLOW_001/results/figures/energy_peak_width.png) · [固定父波形](work/R35_PSD_ORDERED_FLOW_001/results/figures/fixed_parent_boundary_examples.png) · [复现入口](work/R35_PSD_ORDERED_FLOW_001/REPRODUCTION.md) · [R36协议](tasks/R36_CONDITIONAL_DIFFUSION_EDIT_001.md)。

实际私有Git和D副本完成情况见R35_PUBLICATION_RECEIPT.json与R35_BACKUP_RECEIPT.json。同次历史暴露采集的DEV比较不构成独立采集泛化证明。
