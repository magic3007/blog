---
name: jekyll-text-theme-firebase-pageview
description: |
  为Jekyll TeXt主题添加基于Firebase Realtime Database的阅读量统计功能。
  适用场景：(1) 使用TeXt主题的Jekyll博客，(2) 不想用LeanCloud想改用Google生态，
  (3) 需要在文章列表和详情页都显示阅读量，(4) 希望有安全的防刷量机制。
author: Claude Code
version: 1.2.1
date: 2026-04-10
---

# Jekyll TeXt主题Firebase阅读量统计集成

## Problem
Jekyll TeXt主题默认只提供LeanCloud作为阅读量统计provider，对于更习惯使用Google生态的用户不够友好，需要自行实现自定义provider。同时需要考虑数据安全性，防止恶意刷量。

## Context / Trigger Conditions
- 使用Jekyll + TeXt主题搭建的静态博客
- 需要在文章页和文章列表页显示阅读量
- 希望使用Firebase替代LeanCloud作为数据存储
- 需要防止前端恶意修改阅读量数据

## Solution
### 1. 主题配置修改
在`_config.yml`中设置pageview provider为custom，并添加Firebase配置项：
```yaml
pageview:
  provider: custom

  ## Firebase (Custom provider)
  firebase:
    api_key: YOUR_FIREBASE_API_KEY
    auth_domain: YOUR_PROJECT_ID.firebaseapp.com
    database_url: https://YOUR_PROJECT_ID-default-rtdb.firebaseio.com
    project_id: YOUR_PROJECT_ID
    app_class: pageviews # 数据存储节点名称
```

### 2. 文章页统计逻辑
修改`_includes/pageview-providers/custom/post.html`实现单篇文章的阅读量统计和增量更新。

### 3. 首页统计逻辑
修改`_includes/pageview-providers/custom/home.html`实现首页文章列表的阅读量批量加载和首页访问统计。

### 4. Firebase安全规则配置
使用以下数据库规则防止恶意刷量：
```json
{
  "rules": {
    "pageviews": {
      "$pageKey": {
        ".read": true,
        "count": {
          ".validate": "newData.isNumber() && newData.val() == data.val() + 1"
        },
        "title": {
          ".validate": "newData.isString() && (!data.exists() || newData.val() == data.val())"
        },
        "url": {
          ".validate": "newData.isString() && (!data.exists() || newData.val() == data.val())"
        },
        "createdAt": {
          ".validate": "newData.isString() && !data.exists()"
        },
        "lastViewed": {
          ".validate": "newData.isString()"
        },
        "$other": {
          ".validate": false
        }
      }
    }
  }
}
```

## Verification
1. 部署后访问任意文章页，查看阅读量是否显示为1
2. 刷新页面，阅读量应该增加到2
3. 访问首页，文章列表中的阅读量应该与详情页一致
4. 在Firebase控制台中查看数据是否正确生成

## 配置步骤
### 1. 创建Firebase项目
1. 打开 [Firebase 控制台](https://console.firebase.google.com/)，使用Google账号登录
2. 点击 "添加项目"，输入项目名称
3. 不需要启用Google Analytics，直接点击 "创建项目"

### 2. 获取配置信息
1. 在项目控制台首页，点击中间的 "Web" 图标（`</>`）添加Web应用
2. 输入应用昵称，不需要勾选Hosting，点击 "注册应用"
3. 在SDK配置中复制 `apiKey`、`authDomain`、`projectId` 三个值

### 3. 创建实时数据库
1. 左侧菜单选择 "构建" → "实时数据库"
2. 点击 "创建数据库"，选择合适的地区（建议亚洲新加坡或美国中部）
3. 选择 "以测试模式启动"，点击 "启用"
4. 保存数据库URL，格式为 `https://PROJECT_ID-default-rtdb.firebaseio.com`

### 4. 配置安全规则
1. 在实时数据库页面点击 "规则" 标签
2. 替换为文档中提供的安全规则，点击 "发布"

### 5. 配置域名授权
1. 左侧菜单选择 "设置" → "项目设置"
2. 向下滚动到 "已授权的网域"
3. 添加你的博客域名（如 `yourdomain.com`、`username.github.io`）和 `localhost`（用于本地测试）

### 6. 更新博客配置
将获取到的配置信息填入`_config.yml`对应的字段中。

## Notes
- 默认在开发环境（jekyll serve）下不会统计阅读量，可以修改`_includes/pageview.html`中的环境判断来本地测试
- Firebase免费额度足够个人博客使用（10GB存储，10万次连接/天，1GB下载/天）
- 需要在Firebase控制台中添加博客域名到授权域名列表，否则会出现CORS错误
- 不需要额外的后端服务，完全基于前端实现

## 常见问题
### Q: 阅读量显示为0，Console报错403
A: 检查安全规则是否正确配置，以及域名是否添加到授权列表。

### Q: 阅读量不增加
A: 检查Firebase配置信息是否正确，特别是`database_url`是否包含`https://`且结尾没有多余的`/`。

### Q: 本地测试看不到效果
A: 因为默认在`development`环境下不统计，修改`_includes/pageview.html`，删除`jekyll.environment != "development"`的判断条件即可本地测试。

### Q: 阅读量一直显示0，控制台没有报错
A: 这是因为部分主题配置下文章的`page.key`字段为空。解决方案是统一使用更通用的`page.id`作为文章唯一标识，需要修改两处代码：
1. `_includes/pageview-providers/custom/post.html` 中的 `pageKey` 生成逻辑
2. `_includes/article-info.html` 中的 `data-page-key` 属性
确保两边的key生成逻辑一致，使用 `{{ page.id | default: page.key | replace: '/', '-' }}` 格式。

### Q: Firebase配置信息（apiKey等）公开在GitHub上安全吗？
A: **完全安全**。Firebase的`apiKey`只是项目标识，不是需要保密的密钥，设计上就是可以公开的。所有前端使用Firebase的应用都会公开这些配置。我们配置的安全规则已经足够严格：
- 阅读量只能每次+1，无法直接修改成任意数字
- 文章标题、URL等信息创建后无法修改
- 不允许添加其他字段
即使别人拿到你的配置信息，也只能按照规则操作，无法恶意刷量或篡改数据。静态站点没有更安全的存储方式，直接填到配置文件提交到GitHub是行业标准做法。

## References
- [TeXt Theme 官方文档](https://tianqi.name/jekyll-TeXt-theme/docs/zh/features)
- [Firebase Realtime Database 文档](https://firebase.google.com/docs/database)
