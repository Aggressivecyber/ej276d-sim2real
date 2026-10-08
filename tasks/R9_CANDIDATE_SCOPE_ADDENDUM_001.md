# R9 补充协议：纳入用户总体域差距方向的整段映射候选

本补充在R8C任何拟合以及本轮R9读取EVAL之前登记。原 tasks/R9_FROZEN_RETROSPECTIVE_VALIDATION_001.md（SHA256 adb052b6968ce4e478321aff7cec9e274993525edeb34a427e1b32fce54a6860）原字节保留，未变更其已看过的同采集回顾性用途、指标定义、无EVAL调参和最终交付要求。下述候选/视图扩展及预算覆盖原任务对应条目。

用户在R8结束时要求讨论整组真实与仿真数据的总体差距。经R8B只读复审，根代理登记R8C完整波形条件随机映射及单一消融；R8C只用TRAIN/内部验证/DEV，不根据本轮EVAL选配方。因此冻结候选包括历史R3B、R8 JOINT、R8 JOINT_SIGNAL，以及R8C FULL_WAVE_STOCHASTIC和PULSE_STOCHASTIC_OFF的全部已保存有限有效候选。不得按DEV/旧EVAL成绩删掉不利臂。未完成或非有限情况单列，有限前缀模型如参与则明确其实际更新数/终止原因，不冒称完整训练成功。

候选冻结必须绑定这两份R9协议、R8C实际配方/源码、所选权重、主生成和实际随机输入SHA及父顺序。原 CONTINUATION_METHOD_PLAN.json 保留初始登记，另存补充登记及新的 CONTINUATION_FROZEN_METHOD.json。旧 FROZEN_METHOD.json 与R1–R8已封存产物不覆盖。

统一评价固定主视图为完整shape、完整V、prompt、postpeak、tail、baseline（R7已有定义）。最多5候选×6视图=30个HGB，加REAL–REAL及既定阳性对照各6个，合计最多42；不另搜索评价器。原R9的条件组隔离/600训练250留出/500次bootstrap、64次同样本量REAL参照、原hash选择2000与全7583补充、未知全保留、共同支持误差和迁移口径全部继续适用。预算为阶段75分钟、累计数值1800秒、4线程/4GiB/512MiB、响应更新0、SSH0、新G4为0。源控制与已独立通过的模型重放直接引用R8/R8C封存证据；无新冲突则不重复全量重放。

条件宽度比较统一用所有相关域共同有定义的窄电荷区间，并列各模型自己的可定义覆盖/失败；新候选缺口不能让平均误差看似降低。原8窄区间和拟合程序不改，保留原报告使用的密度定义。单位/形状视图、原电压波形带、真正pre-peak区间与10–90% rise、基线统计、source-to-output迁移都与训练阶段相同处理。prompt指标不能改称纯上升沿指标。

R9结果不回流模型/选点/随机种子。若完整波形、条件峰宽或源依赖仍不成立，形成明确的有界负结果及停止结论，说明剩余源谱/混合混杂和独立采集缺口。随后完成 CONTINUATION_REPORT.md、CONTINUATION_REPRODUCTION.md、CONTINUATION_DELIVERY_INDEX.json、Git远端核对及D盘复制后SHA/字节核验，再结束当前持续目标。
