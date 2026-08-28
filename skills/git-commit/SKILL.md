---
name: git-commit
description: 帮助用户编写规范的 Git commit message
---

# Git Commit 规范

当用户请求你帮忙 commit 时，遵循以下规范:

## 提交格式
使用 Conventional Commits 格式:
```
<type>(<scope>): <subject>

<body>

<footer>
```

## 类型 (type)
- feat: 新功能
- fix: 修复bug
- docs: 文档变更
- style: 代码格式变更(不影响代码逻辑)
- refactor: 重构(既不是新增功能也不是修复bug)
- test: 增加测试
- chore: 构建过程或辅助工具的变动

## 示例
```
feat(auth): 添加JWT令牌刷新机制

实现 access token 过期前自动刷新逻辑,
避免用户频繁重新登录。
```

## 规则
1. subject 不超过50个字符
2. subject 使用祈使语气
3. subject 不以句号结尾
4. body 解释 what 和 why,不是 how
5. 如果有必要,在 footer 中引用 issue
```
