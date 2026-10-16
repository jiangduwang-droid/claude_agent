"""自迭代(Self-Evolution)模块。

对齐 Hermes Agent 的自进化闭环:

1. **轨迹采集** —— 每轮对话结束后,把用户输入、工具调用序列、工具结果、
   最终回答组装成一条 :class:`Trajectory` 持久化。
2. **后台评审** —— :class:`EvolutionReviewer` 在流结束后异步评审轨迹:
   - gating:只评审「成功」且「工具调用数达到阈值」的轨迹;
   - 用 LLM 判断该轨迹是否代表一个可复用的问题解决模式;
   - 若是,生成一个新的 SKILL.md 技能并写入技能目录。
3. **技能重载** —— 新技能写入后,通知 SkillRegistry 重新扫描,
   下一轮对话即可被 Agent 通过 ``use_skill`` 发现和加载。

该模块与沙箱、权限等完全解耦,只依赖 LLM、SkillRegistry 与文件系统。
"""
