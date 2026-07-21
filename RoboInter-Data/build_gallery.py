"""Build a diverse DROID/RH20T crossview-mask sample set + an HTML card gallery.

Stages (each resumable — skips work that already exists):
  select  pick ~N diverse bases per dataset (dedupe by language instruction),
          write samples/gallery_manifest.json
  fetch   queued npz extraction via HTTP-Range (skip-existing = resume) +
          single-pass mp4 extraction from the local chunk tars
  render  crossview montages via visualize_robointer_masks + 640px JPEG thumbs
  html    samples/gallery.html (self-contained card page, no external deps)
  all     select + fetch + render + html

Usage:
  python build_gallery.py select [--n 50]
  python build_gallery.py all    [--n 50]
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import random
import sys
import tarfile
import time
from datetime import date
from pathlib import Path

import cv2
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fetch_segmentation_npz as fs  # noqa: E402
import visualize_robointer_masks as viz  # noqa: E402

SAMPLES = ROOT / "samples"
THUMBS = SAMPLES / "thumbs"
MANIFEST = SAMPLES / "gallery_manifest.json"
VAL = set(json.loads((ROOT / "val_video.json").read_text(encoding="utf-8")))


def _instr(parquet_path: Path) -> tuple[str, int]:
    pf = pq.ParquetFile(parquet_path)
    n = pf.metadata.num_rows
    t = pf.read_row_group(0, columns=["annotation.instruction_add"]).to_pandas()
    return str(t.iloc[0, 0]).strip(), n


def _diverse(cands: list[dict], n: int, seed: int = 0) -> list[dict]:
    """One per unique instruction first, then second pass, ... until n."""
    rng = random.Random(seed)
    by_instr: dict[str, list[dict]] = {}
    for c in cands:
        by_instr.setdefault(c["instr"].lower(), []).append(c)
    keys = list(by_instr)
    rng.shuffle(keys)
    out, rounds = [], 0
    while len(out) < n and rounds < 60:
        for k in keys:
            if rounds < len(by_instr[k]):
                out.append(by_instr[k][rounds])
                if len(out) >= n:
                    break
        rounds += 1
    return out


def select(n: int) -> list[dict]:
    entries: list[dict] = []

    print("scanning DROID pairs ...")
    pairs = viz.droid_pairs()
    droid_cands = []
    for base, info in sorted(pairs.items()):
        if len(info) < 2:
            continue
        (i1, n1), (i2, n2) = info["1"], info["2"]
        if not (viz.SEG_MAP.get(n1) and viz.SEG_MAP.get(n2)):
            continue
        instr, frames = _instr(viz.DATASETS["droid"]["chunk_dir"] / f"episode_{i1:06d}.parquet")
        droid_cands.append({
            "ds": "droid", "base": base, "instr": instr, "frames": frames,
            "val": n1 in VAL or n2 in VAL, "eps": [i1, i2], "npz": [n1, n2],
            "fig": f"{base}_crossview_mask.png",
        })
    picked = _diverse(droid_cands, n)
    print(f"DROID: {len(droid_cands)} candidates, {len({c['instr'].lower() for c in droid_cands})} unique instructions -> picked {len(picked)}")
    entries += picked

    print("scanning RH20T takes ...")
    takes = viz.rh20t_takes()
    rh_cands = []
    for take, cams in sorted(takes.items()):
        if len(cams) < 2:
            continue
        cams2 = sorted(cams)[:2]  # must match render_rh20t's choice
        (iA, nA), (iB, nB) = cams2
        instr, frames = _instr(viz.DATASETS["rh20t"]["chunk_dir"] / f"episode_{iA:06d}.parquet")
        short = take.replace("RH20T_cfg1_", "")
        rh_cands.append({
            "ds": "rh20t", "base": short, "take": take, "instr": instr, "frames": frames,
            "val": nA in VAL or nB in VAL, "eps": [iA, iB], "npz": [nA, nB],
            "n_cams": len(cams), "fig": f"{short}_crossview_mask.png",
        })
    picked = _diverse(rh_cands, n)
    print(f"RH20T: {len(rh_cands)} candidate takes, {len({c['instr'].lower() for c in rh_cands})} unique instructions -> picked {len(picked)}")
    entries += picked

    SAMPLES.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"manifest -> {MANIFEST} ({len(entries)} entries)")
    return entries


def fetch(entries: list[dict]) -> None:
    need = []
    for e in entries:
        for name in e["npz"]:
            if not any((d / f"{name}.npz").exists() for d in viz.NPZ_DIRS):
                need.append(name)
    print(f"npz queue: {len(need)} to fetch (skip-existing on)")
    if need:
        index = fs.load_index()
        by_base = {Path(m).name: m for m in index["entries"]}
        for i, name in enumerate(need, 1):
            member = by_base.get(f"{name}.npz")
            if member is None:
                print(f"  !! not in archive: {name}")
                continue
            out = fs.OUT_DIR / f"{name}.npz"
            for attempt in range(3):
                try:
                    fs.extract(index, member, out)
                    print(f"  [{i}/{len(need)}] {name}.npz")
                    break
                except Exception as exc:
                    print(f"  retry {name} ({attempt + 1}/3): {exc}")
                    time.sleep(4)
            else:
                print(f"  !! FAILED {name}")

    for ds in ("droid", "rh20t"):
        cfg = viz.DATASETS[ds]
        wanted = []
        for e in entries:
            if e["ds"] != ds:
                continue
            iA, iB = e["eps"]
            wanted += [
                f"chunk-000/observation.images.primary/episode_{iA:06d}.mp4",
                f"chunk-000/observation.images.primary/episode_{iB:06d}.mp4",
                f"chunk-000/observation.images.wrist/episode_{iA:06d}.mp4",
            ]
        missing = [r for r in dict.fromkeys(wanted) if not (cfg["extract_root"] / r).exists()]
        print(f"{ds}: {len(missing)} mp4 to extract from {cfg['tar'].name}")
        if not missing:
            continue
        with tarfile.open(cfg["tar"]) as tf:
            by_name = {m.name.lstrip("./"): m for m in tf.getmembers() if m.isfile()}
            for r in missing:
                m = by_name.get(r) or next((mm for k, mm in by_name.items() if k.endswith(r)), None)
                if m is None:
                    print(f"  !! {r} not in tar")
                    continue
                m.name = r
                tf.extract(m, cfg["extract_root"])
        print(f"  {ds} extraction done")


def render(entries: list[dict], force: bool = False) -> None:
    pairs = viz.droid_pairs()
    takes = viz.rh20t_takes()
    done = fails = skip = 0
    for e in entries:
        out = SAMPLES / e["fig"]
        if out.exists() and not force:
            skip += 1
            continue
        try:
            if e["ds"] == "droid":
                viz.render_droid(e["base"], pairs)
            else:
                viz.render_rh20t(e["take"], takes)
            done += 1
        except Exception as exc:
            fails += 1
            print(f"  !! render failed {e['base']}: {exc}")
    print(f"render: {done} new, {skip} existing, {fails} failed")

    THUMBS.mkdir(exist_ok=True)
    for e in entries:
        src, dst = SAMPLES / e["fig"], THUMBS / (Path(e["fig"]).stem + ".jpg")
        if not src.exists() or (dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime):
            continue
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        tw = 640
        cv2.imwrite(str(dst), cv2.resize(img, (tw, int(h * tw / w)), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("thumbs done")


def build_html(entries: list[dict]) -> None:
    live = [e for e in entries if (SAMPLES / e["fig"]).exists()]
    n_d = sum(1 for e in live if e["ds"] == "droid")
    n_r = sum(1 for e in live if e["ds"] == "rh20t")
    droid_scenes = {e["scene_id"] for e in live if e.get("scene_id") is not None}
    droid_labs = {e["lab"] for e in live if e.get("lab")}
    rh_instrs = {e["instr"].lower() for e in live if e["ds"] == "rh20t"}

    cards = []
    for e in sorted(live, key=lambda x: (x["ds"], x["base"])):
        thumb = f"thumbs/{Path(e['fig']).stem}.jpg"
        sam2 = viz.load_sam2_wrist(e["base"]) is not None
        badges = f'<span class="b {e["ds"]}">{e["ds"].upper()}</span>'
        if e.get("lab"):
            badges += f'<span class="b lab">{html_mod.escape(str(e["lab"]))}</span>'
        if e.get("scene_id") is not None:
            tip = html_mod.escape(f'scene_id {e["scene_id"]} · {e.get("building") or ""}'.strip())
            badges += f'<span class="b scene" title="{tip}">s{str(e["scene_id"])[-5:]}</span>'
        elif e.get("collect_date"):
            badges += f'<span class="b scene">{e["collect_date"]}</span>'
        if e.get("val"):
            badges += '<span class="b val">VAL</span>'
        if sam2:
            badges += '<span class="b sam2">SAM2 wrist</span>'
        if e.get("n_cams"):
            badges += f'<span class="b cams">{e["n_cams"]} cams</span>'
        cards.append(f'''
<a class="card" data-ds="{e["ds"]}" data-q="{html_mod.escape(e["instr"].lower())} {e["base"]} {html_mod.escape(str(e.get("lab") or "").lower())} {html_mod.escape(str(e.get("building") or "").lower())} {e.get("scene_id") or ""}"
   href="{e["fig"]}" target="_blank">
  <img loading="lazy" src="{thumb}" alt="">
  <div class="body">
    <div class="instr">{html_mod.escape(e["instr"]) or "(no instruction)"}</div>
    <div class="meta"><code>{e["base"]}</code><span>{e["frames"]}f</span></div>
    <div class="badges">{badges}</div>
  </div>
</a>''')

    page = f'''<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboInter cross-view mask gallery</title>
<style>
  :root {{ --bg:#f4f6f9; --panel:#fff; --ink:#1c2028; --mute:#7a8294; --line:#e3e7ee;
          --banner:#181c26; --blue:#2563eb; --purple:#8b5cf6; --orange:#ea580c; --green:#0e9f6e; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--ink); font:15px/1.55 "Segoe UI",system-ui,sans-serif; }}
  header {{ background:var(--banner); color:#fff; padding:26px 32px 22px; }}
  header h1 {{ font-size:24px; letter-spacing:.2px; }}
  header p {{ color:#98a2b8; margin-top:6px; font-size:13.5px; max-width:980px; }}
  .stats {{ margin-top:14px; display:flex; gap:10px; flex-wrap:wrap; }}
  .stat {{ background:#232a3a; border-radius:8px; padding:6px 12px; font-size:13px; color:#cdd5e4; }}
  .stat b {{ color:#fff; }}
  .bar {{ position:sticky; top:0; z-index:5; background:var(--panel); border-bottom:1px solid var(--line);
         padding:10px 32px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  .tab {{ border:1px solid var(--line); background:#fff; border-radius:999px; padding:5px 14px;
         cursor:pointer; font-size:13.5px; color:var(--mute); }}
  .tab.on {{ background:var(--banner); border-color:var(--banner); color:#fff; }}
  #q {{ margin-left:auto; border:1px solid var(--line); border-radius:8px; padding:6px 12px;
       font-size:13.5px; min-width:260px; }}
  main {{ padding:22px 32px 40px; display:grid; gap:18px;
         grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden;
          text-decoration:none; color:inherit; display:flex; flex-direction:column;
          transition:box-shadow .15s, transform .15s; }}
  .card:hover {{ box-shadow:0 8px 24px rgba(24,28,38,.12); transform:translateY(-2px); }}
  .card img {{ width:100%; display:block; aspect-ratio:1270/1062; object-fit:cover; background:#12141a; }}
  .body {{ padding:12px 14px 13px; display:flex; flex-direction:column; gap:8px; }}
  .instr {{ font-weight:600; font-size:14.5px; line-height:1.4; display:-webkit-box;
           -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; min-height:2.8em; }}
  .meta {{ display:flex; gap:10px; align-items:baseline; color:var(--mute); font-size:12.5px; }}
  .meta code {{ font-family:Consolas,monospace; font-size:12px; background:var(--bg);
               padding:1px 6px; border-radius:5px; }}
  .badges {{ display:flex; gap:6px; flex-wrap:wrap; }}
  .b {{ font-size:11px; font-weight:700; letter-spacing:.4px; color:#fff; border-radius:5px; padding:2px 7px; }}
  .b.droid {{ background:var(--blue); }} .b.rh20t {{ background:var(--purple); }}
  .b.val {{ background:var(--orange); }} .b.sam2 {{ background:var(--green); }}
  .b.cams {{ background:#64748b; }} .b.lab {{ background:#0e7490; }} .b.scene {{ background:#475569; }}
  footer {{ color:var(--mute); font-size:12.5px; padding:0 32px 30px; }}
</style></head>
<body>
<header>
  <h1>RoboInter cross-view mask gallery</h1>
  <p>每张卡片 = 一条 DROID / RH20T episode:两路外部/全局相机叠加 RoboInter 官方 GT mask(绿);
     wrist 列无官方标注,橙色为我们用 SAM2 生成的预测(contact-frame 自动提示 + 全程传播,
     少数样例为人工点击;<b>非真值、未人工复核</b>)。点击卡片查看原图。</p>
  <div class="stats">
    <div class="stat">DROID <b>{n_d}</b> 条 · 覆盖场景 ≥<b>{len(droid_scenes)}</b> · lab <b>{len(droid_labs)}</b></div>
    <div class="stat">RH20T <b>{n_r}</b> takes · 指令 <b>{len(rh_instrs)}</b> 种(cfg1 单一平台,多样性有限)</div>
    <div class="stat">生成 <b>{date.today().isoformat()}</b></div>
    <div class="stat">源 <b>InternRobotics/RoboInter-Data</b> · CC BY-NC-SA 4.0</div>
  </div>
</header>
<div class="bar">
  <button class="tab on" data-f="all">全部</button>
  <button class="tab" data-f="droid">DROID</button>
  <button class="tab" data-f="rh20t">RH20T</button>
  <input id="q" placeholder="按指令/编号搜索 …">
</div>
<main id="grid">{"".join(cards)}
</main>
<footer>脚本:build_gallery.py(select→fetch→render→html,断点可续);mask 语义与获取方式见 samples/README.md 与 survey §0.7。
场景徽标:s{{scene_id 后 5 位}} 来自 DROID 官方逐条 metadata(GCS droid_raw),悬停看完整 id 与 building;少数 BVL 条目官方缺 metadata,以采集日期代理(计入 lab 不计入场景数)。</footer>
<script>
  const tabs=[...document.querySelectorAll('.tab')], q=document.getElementById('q');
  let f='all';
  function apply() {{
    const s=q.value.trim().toLowerCase();
    for (const c of document.querySelectorAll('.card')) {{
      const okF = f==='all' || c.dataset.ds===f;
      const okQ = !s || c.dataset.q.includes(s);
      c.style.display = okF && okQ ? '' : 'none';
    }}
  }}
  tabs.forEach(t=>t.onclick=()=>{{tabs.forEach(x=>x.classList.remove('on'));t.classList.add('on');f=t.dataset.f;apply();}});
  q.oninput=apply;
</script>
</body></html>'''
    out = SAMPLES / "gallery.html"
    out.write_text(page, encoding="utf-8")
    print(f"gallery -> {out} ({len(live)} cards)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "fetch", "render", "html", "all"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="re-render even if the figure exists")
    args = ap.parse_args()

    if args.stage == "select" or not MANIFEST.exists():
        entries = select(args.n)
        if args.stage == "select":
            return
    else:
        entries = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if args.stage in ("fetch", "all"):
        fetch(entries)
    if args.stage in ("render", "all"):
        render(entries, force=args.force)
    if args.stage in ("html", "all"):
        build_html(entries)


if __name__ == "__main__":
    main()
