---
name: caching
summary: 为分布式系统选择和设计 CDN、Redis 与缓存一致性策略。
---

# Caching Strategies

## 触发条件与不适用场景

在请求涉及 CDN、热点、缓存、TTL、失效、读性能或数据库读压力时使用。不适用于必须读取绝对最新状态的唯一真相写入边界。

## 输入清单

- 读写比、峰值 QPS、热点分布和数据大小。
- 可接受陈旧时间、失效事件和一致性要求。
- 客户端、边缘、服务端与数据库之间的缓存位置。
- 回源容量、成本、地域和故障恢复目标。

## 决策流程

1. 静态或可边缘缓存内容优先使用 CDN。
2. 高频动态读取且允许短暂陈旧时使用 Redis 或本地缓存。
3. 读多写少默认选择 cache-aside；写后必须立即可读时选择 write-through；可容忍异步持久化风险时才考虑 write-behind。
4. 为每个键定义 TTL、主动失效事件、回源保护与容量指标。

## 权衡与失败模式

- CDN 降低边缘延迟，但会增加失效传播复杂度。
- cache-aside 简单，但存在短暂双写不一致。
- write-through 更一致但增加写延迟；write-behind 吞吐高但可能丢失未落盘数据。
- 缓存穿透使用负缓存或布隆过滤器；击穿使用请求合并或互斥重建；雪崩使用 TTL 抖动和限流；热点键使用副本或分片。

## 案例图

- [Netflix](../../cases/Netflix%20-%20JHNL.excalidraw)
- [Newsfeed](../../cases/Newsfeed%20-%20JHNL.excalidraw)
- [Mock Google Photos](../../cases/Mock%20-%20Google%20Photos.excalidraw)

## 输出模板

```text
缓存位置与内容：
键与分区方式：
读写策略：
TTL 与失效链路：
回源保护与容量：
陈旧性、成本与一致性权衡：
失败模式与监控指标：
参考案例图：
```
