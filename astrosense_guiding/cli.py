"""Command line interface: ``astrosense scan|analyze|synth``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze_session, to_jsonable, usable_sessions
from .parser import parse_file


def _cmd_scan(args) -> int:
    paths = []
    for p in args.paths:
        p = Path(p)
        paths += sorted(p.glob("PHD2_GuideLog*.txt")) if p.is_dir() else [p]
    for p in paths:
        log = parse_file(p)
        good = usable_sessions(log)
        print(f"{p.name}: guiding sessions {len(log.sessions)} (解析可能 {len(good)}), "
              f"calibrations {log.n_calibrations} (star lost {log.n_calibration_star_lost})")
        for s in log.sessions:
            print("   ", s.summary())
    return 0


def _cmd_analyze(args) -> int:
    from .report import plot_report, text_report

    log = parse_file(args.log)
    good = usable_sessions(log)
    if not good:
        print(f"解析可能なガイドセッションがありません（ガイドセッション {len(log.sessions)}、"
              f"キャリブレーション {log.n_calibrations}、うち星喪失 {log.n_calibration_star_lost} 行）。")
        return 2
    if args.session == "all":
        targets = good
    elif args.session is None:
        targets = [max(good, key=lambda s: int(s.valid.sum()))]
    else:
        targets = [s for s in log.sessions if s.index == int(args.session)]
        if not targets:
            print(f"セッション #{args.session} が見つかりません")
            return 2
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.log).stem
    for s in targets:
        a = analyze_session(s, args.worm_period)
        print(text_report(a))
        if out:
            base = out / f"{stem}_s{s.index}"
            Path(f"{base}.json").write_text(json.dumps(to_jsonable(a), ensure_ascii=False, indent=2),
                                            encoding="utf-8")
            Path(f"{base}.txt").write_text(text_report(a), encoding="utf-8")
            if not args.no_plot:
                plot_report(a, f"{base}.png")
            print(f"\n出力: {base}.txt / .json" + ("" if args.no_plot else " / .png"))
    return 0


def _cmd_synth(args) -> int:
    from .synth import SynthConfig, generate

    cfg = SynthConfig(seed=args.seed, duration_s=args.duration, dec_backlash_ms=args.backlash,
                      dither_every_s=args.dither, n_gusts=args.gusts)
    Path(args.out).write_text(generate(cfg), encoding="utf-8")
    truth = cfg.truth()
    Path(args.out).with_suffix(".truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {args.out}  (PE P-P sky {truth['pe_pp_sky']:.2f}\", backlash {cfg.dec_backlash_ms:.0f} ms)")
    return 0


def _cmd_benchmark(args) -> int:
    from .online_eval import (BenchmarkConfig, benchmark_markdown,
                              plot_benchmark, run_benchmark)

    result, examples = run_benchmark(args.seeds, BenchmarkConfig(forgetting=args.forgetting))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'online_benchmark.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report = benchmark_markdown(result)
    (out / 'online_benchmark.md').write_text(report, encoding='utf-8')
    plot_benchmark(result, examples, out / 'online_benchmark.png')
    print(report)
    print(f'出力: {out.resolve()}')
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="astrosense", description="PHD2 guide-log analyzer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("scan", help="ログ/フォルダ内のセッション一覧")
    sp.add_argument("paths", nargs="+")
    sp.set_defaults(func=_cmd_scan)

    sp = sub.add_parser("analyze", help="周期誤差・バックラッシュ・振動を解析")
    sp.add_argument("log")
    sp.add_argument("--session", default=None, help="セッション番号 or 'all'（既定: 最長）")
    sp.add_argument("--worm-period", type=float, default=None, help="ウォーム周期 [s] を指定")
    sp.add_argument("--out", default=None, help="レポート出力フォルダ")
    sp.add_argument("--no-plot", action="store_true")
    sp.set_defaults(func=_cmd_analyze)

    sp = sub.add_parser("synth", help="検証用の合成 PHD2 ログを生成")
    sp.add_argument("out")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--duration", type=float, default=3600.0)
    sp.add_argument("--backlash", type=float, default=800.0, help="Dec バックラッシュ [ms]")
    sp.add_argument("--dither", type=float, default=600.0, help="ディザ間隔 [s] (0=なし)")
    sp.add_argument("--gusts", type=int, default=3)
    sp.set_defaults(func=_cmd_synth)

    sp = sub.add_parser("benchmark", help="合成データで固定PECとオンラインRLSを比較")
    sp.add_argument("--out", default="reports/online", help="評価結果の出力先")
    sp.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 6)))
    sp.add_argument("--forgetting", type=float, default=0.995)
    sp.set_defaults(func=_cmd_benchmark)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
