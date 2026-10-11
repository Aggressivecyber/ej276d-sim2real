# R69 后处理入口：独立静态复核

结论：当前完整比较路径无阻断。root从保存的EVALUATION独立重算原门，再核对RESULT；本次发现的两处复核缺口已补齐。原A–F与完整模型目标仍未完成。

只读取review_and_report.py及冻结pilot.py源码；未执行任一脚本，未打开结果科学数组/模型权重、SSH或修改冻结文件。下述行号对应本次封存源码。

- 主门独立重算：review_and_report.py:55–69读取baseline/candidate/mini EVALUATION，核对8个共同seed、6593/21047人口、九项SOURCE/五项REAL-control pieces、score总和、电荷误差和全部原PSD区域。83–88从这些保存值重新计算两项10%收益、至少6/8胜出及配对噪声bootstrap；不是把RESULT里的pass_gate当答案。
- 102项完整联合门：89–109从EVALUATION重算SOURCE四组及REAL两组所有阈值，检查102项并与RESULT逐项比较。相关门、谷门、非负项门均与pilot.py:382–397一致；unknown失败。候选新键candidate映射原RESULT旧键full正确（review:107）。原region_-1、region_0..7、region_8及总体保留（69,111–112）；冻结pilot的原schema为SOURCE59+REAL43。主门与所有harm门仍取AND。
- 抽样曝光：review:70–82按原PCG64 seed重建100次、每次4/104无放回原块选择，核对保存序列与6593父反传次数，末1父按min截断保留；核对400次VJP、实际forward行数及candidate/mini counter_delta。这里只审阅验证方法，没有打开NPZ。原参考梯度零更新、104块及方差在29–38另有独立复算/计数检查。
- 资源及时间：review:23–25核对12 CPU、native RSS及GPU进程峰值；40–43已补查已完成/负准入路径的worker elapsed<600和GPU allocated/reserved上限。57–59保留candidate100步、两臂<=200秒及原等时容差/LR。native supervisor<=601秒与worker<600秒分别核对，不混为一个上限。
- 不完整运行计数：review:156–158,175已修正；completed记2，负准入/预算前止步记0，其余不完整状态记unknown，并另列已完成训练收据数，不再把可能已训练的执行失败记成0次启动。

两项修复由root完成，本审核未改其代码。本静态意见不代替root实际后处理exit0、数组重算、原生资源收据或最终交付；root另行报告的NEGATIVE_LOCAL_SUPPORT也不能推出原固定A–F普遍不可行。

当前输入SHA256：

- review_and_report.py：52dc85fd8d5a3502c511044155ae566ff72834c004942d33bdfbfa52a79e12ca
- pilot.py：47bd6eb5256d7c376ad60af2ebd69e79265fd1cf64384021c97eb1da23bd31c4

自身1 CPU（mask4）、RSS硬上限1 GiB，全部新增文件/缓存在E盘本审核目录。输入哈希和资源/写入退出证据见POSTPROCESS_INPUT_HASH_RECEIPT.json、POSTPROCESS_RESOURCE_RECEIPT.json。
