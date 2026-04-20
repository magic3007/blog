# Firebase Realtime Database Pageview Always Showing 0

**日期**: 2026-04-21
**相关组件/模块**: Firebase Realtime Database, Jekyll Pageview Provider
**状态**: 已解决

## 问题描述
Jekyll博客的Firebase Realtime Database pageview功能部署后，所有页面的阅读量始终显示为0。

## 症状与错误信息
- 所有文章页面和首页文章列表的阅读量显示为"0"
- 浏览器控制台无明显报错（因为Firebase SDK模块加载失败时静默失败）
- Firebase REST API `GET /pageviews.json` 返回 `null`（数据库为空）
- Firebase REST API `PUT` 写入返回 `401 Permission denied`

## 根本原因分析

**两层问题叠加**：

### 问题1：Firebase安全规则未正确部署
- `firebase.json` 只配置了 `hosting`，没有 `database` 部分
- 运行 `firebase deploy` 只部署了静态站点，**不会更新数据库安全规则**
- 在Firebase Console网页界面修改规则可能也没有生效（需通过CLI部署）
- 默认安全规则 `".read": false, ".write": false` 拒绝所有访问

### 问题2：count字段的validate规则过于严格
- 初始提供的validate规则 `newData.val() === data.val() + 1` 在 `data.val()` 为 `null`（新记录）时，`null + 1` 的行为在Firebase规则引擎中导致验证失败
- 即使逻辑上有 `||` 分支处理新记录场景，整个表达式仍被拒绝
- 通过REST API测试确认：写入 `title`（简单validate）成功，写入 `count`（复杂validate）失败

## 解决方案

### 步骤1：在firebase.json中添加database配置
```json
{
  "database": {
    "rules": "database.rules.json"
  },
  "hosting": { ... }
}
```

### 步骤2：创建简化的安全规则文件 database.rules.json
```json
{
  "rules": {
    "pageviews": {
      ".read": true,
      "$page": {
        ".write": true
      }
    }
  }
}
```

### 步骤3：使用firebase CLI部署规则
```bash
firebase deploy --only database
```

### 步骤4：简化post.html中的计数逻辑
- 移除所有debug console.log
- 使用 `increment(1)` 原子操作替代 get-then-update 模式
- 消除并发竞态条件

## 验证方法
```bash
# 1. 测试读取权限
curl -s "https://<project>.firebaseio.com/pageviews.json"
# 期望: 200, null 或实际数据

# 2. 测试写入权限
curl -s -w "%{http_code}" -X PATCH \
  "https://<project>.firebaseio.com/pageviews/_test.json" \
  -d '{"count":1,"title":"Test"}'
# 期望: 200

# 3. 清理测试数据
curl -s -X DELETE "https://<project>.firebaseio.com/pageviews/_test.json"
```

## 关键学习经验
1. **`firebase deploy` 不等于部署所有服务** — 如果 `firebase.json` 没有配置 `database` 部分，数据库规则不会被部署
2. **Firebase规则中的null算术** — `data.val() + 1` 当 `data.val()` 为 null 时，即使有 OR 分支处理，复杂的validate表达式也可能被Firebase规则引擎拒绝
3. **用REST API逐字段测试** — 通过 `curl` 直接测试不同字段的写入，可以快速定位是哪个validate规则在阻止写入
4. **Jekyll环境变量** — `jekyll serve` 默认是 `development` 环境，pageview脚本在此环境下不会加载。需要 `JEKYLL_ENV=production` 才能本地测试

## 预防措施
1. 新增Firebase服务时，确认 `firebase.json` 包含对应的服务配置
2. Firebase安全规则从简单开始，确认基本读写功能后再逐步添加验证
3. 使用REST API curl测试验证规则变更是否生效，不要只依赖Console界面
4. 将 `database.rules.json` 纳入版本控制，确保规则可追溯

## 相关资源
- `firebase.json` — Firebase项目配置
- `database.rules.json` — Realtime Database安全规则
- `_includes/pageview-providers/custom/post.html` — 文章页pageview脚本
- `_includes/pageview-providers/custom/home.html` — 首页pageview脚本
- `_includes/pageview.html` — pageview入口（含环境判断）
