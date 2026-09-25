---
name: system-design
summary: 将架构问题路由到系统设计流程、基础设施和领域模式技能。
---

# System Design

在用户要求设计、评估或改进分布式系统架构时使用。

## 路由

1. 先判断请求是否需要完整设计：新系统设计、架构评审、扩容或多组件改造均属于完整设计，先读取 [process/SKILL.md](process/SKILL.md)。
2. 对单项能力选型，直接加载最少必要的子技能；不要求完整流程。
3. 跨能力问题组合加载多个子技能，不能因单一分类遗漏相关能力。
4. 输出始终列出采用与放弃的方案、关键权衡、主要风险和相关案例图路径。

| 请求特征 | 加载技能 |
|---|---|
| 缓存、CDN、热点与失效 | [infrastructure/caching/SKILL.md](infrastructure/caching/SKILL.md) |
| 数据模型、复制、分区、分片 | [infrastructure/database-design/SKILL.md](infrastructure/database-design/SKILL.md) |
| 队列、事件、CDC、流处理 | [infrastructure/messaging-streaming/SKILL.md](infrastructure/messaging-streaming/SKILL.md) |
| 聊天、推送、长连接、在线状态 | [infrastructure/realtime-communication/SKILL.md](infrastructure/realtime-communication/SKILL.md) |
| 库存、席位、预留、超卖、幂等 | [infrastructure/distributed-concurrency/SKILL.md](infrastructure/distributed-concurrency/SKILL.md) |
| 附近、定位、地图、路线 | [infrastructure/geospatial/SKILL.md](infrastructure/geospatial/SKILL.md) |
| 全文搜索、索引、召回、排序 | [patterns/search-indexing/SKILL.md](patterns/search-indexing/SKILL.md) |
| 上传、对象存储、转码、CDN 分发 | [patterns/file-media/SKILL.md](patterns/file-media/SKILL.md) |
| 邮件、短信、Push、通知偏好 | [patterns/notifications/SKILL.md](patterns/notifications/SKILL.md) |
| 支付、Webhook、外部 API、降级 | [patterns/third-party-integration/SKILL.md](patterns/third-party-integration/SKILL.md) |
| 动态流、关注关系、读写扇出 | [patterns/feed-fanout/SKILL.md](patterns/feed-fanout/SKILL.md) |
| 分享、租户、资源权限、撤销 | [patterns/access-control/SKILL.md](patterns/access-control/SKILL.md) |

## 组合规则

- 聊天应用且缓存热门会话：`process`、`realtime-communication`、`database-design`、`caching`。
- 高并发票务：`process`、`distributed-concurrency`、`database-design`、`third-party-integration`。
- 位置搜索：`process`、`geospatial`、`search-indexing`、`caching`。
- 媒体分享：`process`、`file-media`、`access-control`、`messaging-streaming`、`caching`。

## 输出格式

```text
问题边界
采用方案
放弃方案与原因
关键数据流
一致性、性能、可用性、成本权衡
失败模式与缓解措施
待验证假设
参考案例图
```
