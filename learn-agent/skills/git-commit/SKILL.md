---
name: git-commit
description: 帮助用户编写规范的 Git commit message
---

# Git Commit 规范

当用户请求你帮忙写 commit message 时，遵循以下规范：

## 提交格式
使用 Conventional Commits 格式：

```
<type>(<scope>): <subject>

<body>

<footer>
```

## 类型 (type)
- feat: 新功能
- fix: 修复 bug
- docs: 文档变更
- style: 代码格式变更（不影响逻辑）
- refactor: 重构
- test: 增加测试
- chore: 构建 / 辅助工具变更

## 规则
1. subject 不超过 50 个字符
2. 使用祈使语气，不以句号结尾
3. body 解释 what 和 why，而不是 how
4. 必要时在 footer 引用 issue
