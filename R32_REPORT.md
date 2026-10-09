# R32：均值收益未通过完整响应验收

固定系数1的条件均值配方关闭。两个同服务器2950步末点、同原始94400条训练流比较，共同观察Q区1–7的完整480点平均波形MSE降低48.07%，但完整波形可分性没有可信改善，PSD距离及高侧峰宽误差恶化。

|同2950步末点|Control|Mean|判断|
|---|---:|---:|---|
|完整shape493 BA|.978|.976|差−.002，95%区间[−.020018,.016422]|
|纯原始u480 BA|.978|.978|差0，95%区间[−.017929,.017956]|
|共同Q区均值MSE|.000168769|.000087641|降低48.07%|
|总体PSD W1|.0230655|.0271266|增加17.61%|
|共同Q区PSD W1|.0347658|.0422889|增加21.64%|
|低侧raw-width MAE|.0103927|.0090538|降低12.88%|
|高侧raw-width MAE|.0130512|.0169147|增加29.60%|
|baseline RMS W1 (V)|.0000193584|.0000184454|降低4.72%|
|ACF1 W1|.0642350|.0477606|降低25.65%|

原J20所选端点分别为750与1750步，所选shape493 BA .998/.990、纯u480 .996/.988；不以DEV改选。均值项仅使用REAL TRAIN-fit固定Q区均值及原TRAIN尺度，两生成位置均保留梯度，24个有限负值批原样保留。两臂各2950次G/D/reader更新、实际暴露3752个源父和18739条REAL TRAIN行；6593/21047是载入池大小，不冒充实际暴露数。小张量代数与梯度核对通过，不把当前训练历史样本称为理想iid无偏。

四个端点各2000条原父记录、417 pureEE/1518 pure-had/65 mixed、顺序与权重保留，QC全通过、ADC无裁剪。共同PSD区1–7、可比较密度宽度区2–7；低支持/不分离槽保持undefined。500区组区间条件于固定分类器和当前组，不含训练种子、平台或独立采集不确定性。REAL仍为单次历史暴露采集，两密度成分不是粒子真值。

源Q口径的具体冲突已核清：历史SOURCE_CLEAN是实际未量化NN输入的14bit ADC结果，两份历史参考与该变换逐值相同，父ID不变、无裁剪。量化使23父换Q区，实际source-Q bins3/4的pure-had为213/181，历史postADC参考为207/182。两种条件均单列；实际输入分组的谷比例从5.6338%/.5525%降至.4695%/0，经验SD也收窄，不能据此宣称类别保真提升。元数据full512解析电荷与480裁窗100:400电荷是不同积分对象。

首次评价在汇总时因误把实际preADC源作为历史SOURCE_CLEAN而断言失败；保存原native1和73.8223秒成本。修复只恢复历史参考并组装已保存的40次HGB、32描述槽（实际29次EM）与输出，没有重训/重生成/重拟合；实际输入source-Q描述另补。一次传输控制SyntaxError也保留，原PID未记录为unknown。

科学累计345.523253/2400秒，训练体共224.347971秒。服务器真实worker23246/native0、峰RSS 2,710,675,456字节、GPU allocated 276,043,264字节；2次RSS查询失败的时间与原因unknown，自报峰与观测峰一致，不能将缺测写成0。独立CPU8000条主输出量化前后、signal/electronic及父顺序全部bitwise回放通过。这是模型推理复现，不是跨平台训练一致性证明。执行器实际检查16图，根检查上述八张关键图，封存389文件/238,952,600字节。

下一轮R33已登记：冻结本轮Control2950生成器，以完整源驱动输出为480维流初态；同数据、时刻流、网络和2950更新/900秒上限，比较电荷区内独立CFM和OT-CFM。只用TRAIN共用仿射常数，固定末点及32步Heun，完整波形/PSD/宽度/噪声、源关联和数值稳定共同验收；运输耦合不当作物理配对。R28仍是历史开发比较基线，R33不是已证明成功的方法。

[完整报告](work/R32_CONDITIONAL_MEAN_TRANSLATION_001/REPORT.md) · [均值/SD图](work/R32_CONDITIONAL_MEAN_TRANSLATION_001/results/figures/conditional_SD.png) · [完整分类图](work/R32_CONDITIONAL_MEAN_TRANSLATION_001/results/figures/c2st.png) · [峰宽图](work/R32_CONDITIONAL_MEAN_TRANSLATION_001/results/figures/energy_peak_width.png) · [源Q语义](work/R32_CONDITIONAL_MEAN_TRANSLATION_001/results/figures/source_condition_semantics.png) · [复现入口](work/R32_CONDITIONAL_MEAN_TRANSLATION_001/REPRODUCTION.md) · [R33任务](tasks/R33_RESIDUAL_OT_FLOW_001.md)。

项目active，完整响应尚未验收。实际私有Git和正常D副本回执分别为R32_PUBLICATION_RECEIPT.json与R32_BACKUP_RECEIPT.json。
