#!/usr/bin/env bash
#
# install.sh — 将本仓库的 skills 安装/恢复到目标 ~/.agents
#
# 用法:
#   ./install.sh                安装到默认 ~/.agents(覆盖式,先备份)
#   ./install.sh --target DIR   安装到指定目录(默认 ~/.agents)
#   ./install.sh --dry-run      只预览将要做什么,不实际写入
#   ./install.sh --no-lock      不覆盖目标 .skill-lock.json
#   ./install.sh --no-backup    跳过安装前的备份
#
# 行为:
#   - skills/      → rsync 合并到 <target>/skills/(保留目标已有、仓库没有的技能)
#   - .skill-lock.json → 备份后覆盖(除非 --no-lock)
#   - 若目标无 .skill-lock.json,直接写入
set -euo pipefail

TARGET="${HOME}/.agents"
DRY_RUN=0
NO_LOCK=0
NO_BACKUP=0

# 仓库根目录(脚本所在位置的上一级)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '\033[36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)   TARGET="${2:?--target 需要目录参数}"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --no-lock)  NO_LOCK=1; shift ;;
    --no-backup) NO_BACKUP=1; shift ;;
    -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "未知参数: $1 (用 --help 查看用法)" ;;
  esac
done

SRC_SKILLS="${REPO_ROOT}/skills"
SRC_LOCK="${REPO_ROOT}/.skill-lock.json"

[[ -d "${SRC_SKILLS}" ]]   || die "仓库缺少 skills/ 目录: ${SRC_SKILLS}"
[[ -f "${SRC_LOCK}" ]]     || die "仓库缺少 .skill-lock.json: ${SRC_LOCK}"

log "目标目录: ${TARGET}"
log "仓库技能数: $(find "${SRC_SKILLS}" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ')"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "DRY-RUN 模式,仅预览:"
  log "  rsync -a --delete ${SRC_SKILLS}/ → ${TARGET}/skills/"
  [[ "${NO_LOCK}" -eq 1 ]] || log "  cp ${SRC_LOCK} → ${TARGET}/.skill-lock.json (备份后)"
  exit 0
fi

[[ -d "${TARGET}" ]] || mkdir -p "${TARGET}"

# 1) 备份当前 lock(若存在)
if [[ -f "${TARGET}/.skill-lock.json" && "${NO_BACKUP}" -eq 0 ]]; then
  BAK="${TARGET}/.skill-lock.json.bak.$(date +%Y%m%d%H%M%S)"
  cp "${TARGET}/.skill-lock.json" "${BAK}"
  log "已备份原锁文件 → ${BAK}"
fi

# 2) 合并 skills 目录
mkdir -p "${TARGET}/skills"
rsync -a --delete "${SRC_SKILLS}/" "${TARGET}/skills/"
log "skills/ 已合并 → ${TARGET}/skills/ ($(find "${TARGET}/skills" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ') 个目录)"

# 3) 覆盖 lock
if [[ "${NO_LOCK}" -eq 1 ]]; then
  warn "--no-lock: 未写入 .skill-lock.json"
else
  cp "${SRC_LOCK}" "${TARGET}/.skill-lock.json"
  log ".skill-lock.json 已写入 → ${TARGET}/.skill-lock.json"
fi

log "完成。"