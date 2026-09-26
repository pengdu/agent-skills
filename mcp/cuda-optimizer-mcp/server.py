# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0,<2"]
# ///
"""CUDA Optimizer MCP Server.

Deterministic profiling toolchain for CUDA performance optimization.
Wraps ncu / nsys / analysis scripts as MCP tools. Decision-making
(code changes) stays with the calling model.

Run: uv run server.py
"""

import os
import shutil
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("cuda-optimizer")

SKILL_DIRS = [
    Path.home() / ".agents/skills",
    Path(os.environ.get("CUDA_OPT_SKILLS_DIR", "")) if os.environ.get("CUDA_OPT_SKILLS_DIR") else None,
]

PLAYBOOKS = {
    "auto-tune": "cuda-auto-tune/SKILL.md",
    "streams-concurrency": "cuda-best-practices/references/streams-concurrency.md",
    "memory-optimization": "cuda-best-practices/references/memory-optimization.md",
    "kernel-tuning": "cuda-best-practices/references/kernel-tuning.md",
}


def _find_skill_file(rel: str) -> Path | None:
    for base in SKILL_DIRS:
        if base and (base / rel).exists():
            return base / rel
    return None


def _run(cmd: list[str] | str, workdir: str | None, timeout: int = 1800, shell: bool = False) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, shell=shell, cwd=workdir or None, timeout=timeout,
            capture_output=True, text=True,
        )
        out = (p.stdout or "") + ("\n--- stderr ---\n" + p.stderr if p.stderr else "")
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {timeout}s: {cmd}"
    except FileNotFoundError as e:
        return -1, f"NOT FOUND: {e}"


def _tail(text: str, max_chars: int = 20000) -> str:
    return text if len(text) <= max_chars else "...(truncated)...\n" + text[-max_chars:]


@mcp.tool()
def check_env(workdir: str = "") -> str:
    """Check that ncu / nsys / GPU are available on this machine. Call this first."""
    lines = []
    for tool in ("ncu", "nsys", "nvidia-smi", "python3"):
        path = shutil.which(tool)
        lines.append(f"{tool}: {path or 'NOT FOUND'}")
    if shutil.which("nvidia-smi"):
        rc, out = _run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"], workdir or None, 30)
        lines.append(out.strip())
    profile_sh = _find_skill_file("cuda-auto-tune/scripts/ncu_profile.sh")
    analyse_py = _find_skill_file("cuda-auto-tune/scripts/ncu_analyse.py")
    lines.append(f"ncu_profile.sh: {profile_sh or 'NOT FOUND'}")
    lines.append(f"ncu_analyse.py: {analyse_py or 'NOT FOUND'}")
    return "\n".join(lines)


@mcp.tool()
def profile_kernel(command: str, prefix: str = "report_v1", workdir: str = "", extra_ncu_args: str = "") -> str:
    """Profile CUDA kernels with NCU (--set full) and run deep analysis.

    Args:
        command: kernel binary or command, e.g. "./my_kernel" or "python kernel.py"
        prefix: output report prefix, e.g. "report_v1"
        workdir: working directory to run in (where ncu_reports/ will be created)
        extra_ncu_args: extra ncu args, e.g. "--kernel-name triton_ --launch-count 1"

    Returns the analysis report. Outputs land in <workdir>/ncu_reports/<prefix>.*
    """
    profile_sh = _find_skill_file("cuda-auto-tune/scripts/ncu_profile.sh")
    if not profile_sh:
        return "ERROR: ncu_profile.sh not found. Sync cuda-auto-tune skill to ~/.agents/skills/ or set CUDA_OPT_SKILLS_DIR."
    cmd = f'bash "{profile_sh}" "{command}" "{prefix}" {extra_ncu_args}'.strip()
    rc, out = _run(cmd, workdir or None, timeout=3600, shell=True)
    result = [f"exit={rc}", _tail(out, 8000)]
    analysis = Path(workdir or ".") / "ncu_reports" / f"{prefix}_analysis.md"
    if analysis.exists():
        result.append("\n===== ANALYSIS REPORT =====\n" + _tail(analysis.read_text(), 20000))
    else:
        result.append("\nWARNING: analysis report not generated; check log above.")
    return "\n".join(result)


@mcp.tool()
def analyze_report(csv_path: str, kernel_type: str = "", diff_baseline_csv: str = "") -> str:
    """Analyze an existing NCU CSV report (roofline, memory hierarchy, stalls, occupancy).

    Args:
        csv_path: path to <prefix>.csv exported from NCU
        kernel_type: optional override: cuda | cutlass | triton | cutedsl
        diff_baseline_csv: optional baseline CSV to diff against (iteration comparison)
    """
    analyse_py = _find_skill_file("cuda-auto-tune/scripts/ncu_analyse.py")
    if not analyse_py:
        return "ERROR: ncu_analyse.py not found. Sync cuda-auto-tune skill to ~/.agents/skills/ or set CUDA_OPT_SKILLS_DIR."
    if not Path(csv_path).exists():
        return f"ERROR: {csv_path} not found."
    cmd = ["python3", str(analyse_py), csv_path]
    if kernel_type:
        cmd += ["--type", kernel_type]
    if diff_baseline_csv:
        if not Path(diff_baseline_csv).exists():
            return f"ERROR: baseline {diff_baseline_csv} not found."
        cmd += ["--diff", diff_baseline_csv]
    rc, out = _run(cmd, str(Path(csv_path).parent), 300)
    return f"exit={rc}\n{_tail(out)}"


@mcp.tool()
def profile_system(command: str, prefix: str = "nsys_v1", workdir: str = "") -> str:
    """Profile system-level concurrency with nsys: stream overlap, copy-compute
    pipelining, transfer vs compute time, launch overhead.

    Args:
        command: full application command, e.g. "./app arg1"
        prefix: output report prefix
        workdir: working directory

    Returns nsys stats (kernel summary, memcpy summary, OS runtime). Use this
    for multi-stream concurrency questions; use profile_kernel for single-kernel depth.
    """
    if not shutil.which("nsys"):
        return "ERROR: nsys not found on this machine."
    wd = Path(workdir or ".")
    (wd / "nsys_reports").mkdir(exist_ok=True)
    rep = f"nsys_reports/{prefix}"
    rc, out = _run(f'nsys profile -o "{rep}" --force-overwrite true {command}', str(wd), 3600, shell=True)
    if rc != 0:
        return f"profile failed exit={rc}\n{_tail(out, 8000)}"
    sections = [f"profile ok -> {rep}.nsys-rep"]
    for report in ("cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum", "cuda_api_sum"):
        rc2, out2 = _run(f'nsys stats --report {report} "{rep}.nsys-rep"', str(wd), 300, shell=True)
        sections.append(f"\n===== {report} =====\n{_tail(out2, 6000)}")
    sections.append(
        "\nHints: overlap check = compare total wall time vs (memcpy time + kernel time); "
        "high cudaMemcpy (sync) time in cuda_api_sum suggests pageable memory or missing async; "
        "open the .nsys-rep in nsys-ui to inspect per-stream timeline overlap."
    )
    return "\n".join(sections)


@mcp.tool()
def get_playbook(topic: str) -> str:
    """Get an optimization playbook / best-practices reference.

    Args:
        topic: one of:
          "auto-tune"            - NCU-driven iteration loop + per-kernel-type playbooks (CUDA/CUTLASS/Triton/CuTe DSL)
          "streams-concurrency"  - multi-stream overlap, pinned memory, events, CUDA Graphs, troubleshooting checklist
          "memory-optimization"  - transfers, coalescing, shared memory, bank conflicts, unified memory
          "kernel-tuning"        - occupancy, ILP, divergence, warp primitives, atomics
    """
    rel = PLAYBOOKS.get(topic)
    if not rel:
        return f"ERROR: unknown topic '{topic}'. Valid: {', '.join(PLAYBOOKS)}"
    f = _find_skill_file(rel)
    if not f:
        return f"ERROR: {rel} not found under {[str(d) for d in SKILL_DIRS if d]}."
    return f.read_text()


@mcp.tool()
def clear_jit_cache(kernel_type: str, workdir: str = "") -> str:
    """Clear JIT compilation caches before re-profiling (required after code changes).

    Args:
        kernel_type: "triton" or "cutedsl"
        workdir: project directory (for cutedsl local caches)
    """
    removed = []
    if kernel_type == "triton":
        target = Path.home() / ".triton/cache"
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target))
    elif kernel_type == "cutedsl":
        for name in ("__pycache__", ".cache"):
            target = Path(workdir or ".") / name
            if target.exists():
                shutil.rmtree(target)
                removed.append(str(target))
        tmp = Path("/tmp/cutlass_cute_cache")
        if tmp.exists():
            shutil.rmtree(tmp)
            removed.append(str(tmp))
    else:
        return f"ERROR: unknown kernel_type '{kernel_type}'. Valid: triton, cutedsl"
    return "removed: " + (", ".join(removed) if removed else "(nothing to clear)")


def main():
    mcp.run()


if __name__ == "__main__":
    main()

