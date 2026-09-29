# agent-skills

从 `~/.agents/skills`(以及配套 `.skill-lock.json`)抽取的本地技能库快照,用于版本管理与备份。

## 内容

- `skills/` — 686 个技能目录(每个含 `SKILL.md`),多数来自 skills.sh 生态各仓库,由 Vercel Labs `skills` CLI(`npx skills`)管理安装;少数为本地自撰。
- `.skill-lock.json` — 技能锁文件(v3),记录每个技能的来源仓库、路径、目录哈希、安装/更新时间,是 `skills` CLI 同步的依据。仅覆盖 218 个由 CLI 安装的技能,其余目录(含本地自撰的)不在其中,靠本仓库快照保存。

## 恢复方式

将本仓库内容放回 `~/.agents/` 后,手动还原技能目录:

```bash
# 仅技能目录(不覆盖锁文件以外的配置)
rsync -a --delete skills/ ~/.agents/skills/
```

或按 `.skill-lock.json` 逐条重装:

```bash
npx skills add <owner/repo>@<skill> -g -y
```

## 说明

- `skills/excalidraw-diagram-skill` 原目录含一个嵌套 `.git`,已剥离(保留技能内容)。
- `skills/antigravity-quota` 因 `check-quota.js` 内嵌真实 Google OAuth 凭据,为通过 GitHub push protection 已从仓库历史中移除(磁盘 `~/.agents` 不受影响,仍可用)。
- 快照仅包含技能代码与素材;`video-podcast-maker`、`drawio` 等技能的自带素材体积较大(共约 122M)。
- `skills/cuda-tune-without-ncu` 为本地自撰:Nsight Compute 无法采集计数器时的 CUDA 优化流程(实测带宽微基准、读写消融、SASS/ptxas 静态证据、occupancy API 取代 NCU 指标)。配套改动:`skills/cuda-auto-tune/SKILL.md` 的 GATE 处加了指向它的出口。两者均不在 `.skill-lock.json` 中,只存在于本仓库。