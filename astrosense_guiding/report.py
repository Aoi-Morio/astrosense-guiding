"""Text and figure reports (Japanese)."""
from __future__ import annotations

import math

import numpy as np

from .analysis import Analysis


def _f(x, fmt=".2f", unit=""):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:{fmt}}{unit}"


def recommendations(a: Analysis) -> list[str]:
    rec = []
    pe, bl, vib = a.pe, a.backlash, a.vibration
    if pe.ok and pe.pp_sky is not None:
        if pe.pp_sky > 10:
            rec.append("周期誤差が大きめです。EQMOD/PECPrep 等で PEC 学習、または PHD2 の"
                       "「Predictive PEC」アルゴリズム（RA）の利用を推奨します。")
        if pe.peaks and any(p.label == "非ウォーム成分" and p.snr > 8 for p in pe.peaks):
            rec.append("ウォーム周期以外の強い周期成分があります。モーターギア／ベルトの点検を推奨します。")
        if pe.drift_rate_as_per_min is not None and abs(pe.drift_rate_as_per_min) > 1.0:
            rec.append("RA に一定方向のドリフトがあります（極軸・追尾レート誤差の可能性）。")
    if bl.ok and bl.backlash_ms is not None and bl.backlash_ms > 300 and bl.confidence != "low":
        rec.append(f"Dec バックラッシュ約 {bl.backlash_ms:.0f} ms。ギア調整、意図的な極軸ずらし＋"
                   f"一方向ガイド、または PHD2 バックラッシュ補正（初期値 {0.8*bl.backlash_ms:.0f} ms 程度）を検討してください。")
    if vib.oscillation_ra:
        rec.append("RA が過補正で振動しています。RA Aggression を下げてください。")
    if vib.oscillation_dec:
        rec.append("Dec が過補正で振動しています。Dec Aggression を下げるか Min Move を上げてください。")
    if vib.burst_time_fraction > 0.05:
        rec.append("突発的な外乱が多いです。風よけ・ケーブル固定・三脚の安定化を検討してください。")
    if a.pec_eval.get("ok") and a.pec_eval["improvement"] > 0.1:
        rec.append(f"予測制御（PEC フィードフォワード）で RMS が約 {a.pec_eval['improvement']*100:.0f}% 改善する見込みです。")
    if not rec:
        rec.append("大きな問題は検出されませんでした。")
    return rec


def text_report(a: Analysis) -> str:
    s, ol, pe, bl, vib, ev = a.session, a.openloop, a.pe, a.backlash, a.vibration, a.pec_eval
    p = s.params
    L = []
    w = L.append
    w("=" * 64)
    w(" AstroSense Guiding 解析レポート")
    w("=" * 64)
    w(f"セッション : {s.summary()}")
    w(f"マウント   : {p.get('mount', '—')}   プロファイル: {p.get('profile', '—')}")
    w(f"ピクセル   : {_f(p.get('pixel_scale'), '.2f', ' \"/px')}   Dec: {_f(s.dec_deg, '.1f', '°')}"
      f"   セグメント数(ディザ等): {ol.n_segments}")
    w(f"ガイドレート: RA {ol.ra_plant.rate_px_s:.3f} px/s ({ol.ra_plant.rate_source}), "
      f"Dec {ol.dec_plant.rate_px_s:.3f} px/s ({ol.dec_plant.rate_source})")
    w("")
    w("■ 周期誤差 (RA)")
    if pe.ok:
        w(f"  ウォーム周期 : {pe.worm_period:.1f} s  [{pe.worm_source}]  ログ長 {pe.cycles_covered:.2f} 周期")
        w(f"  P-P (天球上) : {_f(pe.pp_sky, '.2f', ' \"')}   RA軸換算: {_f(pe.pp_axis, '.2f', ' \"')}")
        for h in pe.harmonics:
            w(f"    第{h['n']}高調波 {h['period']:7.1f} s  振幅 ±{h['amplitude']:.2f}\"")
        w(f"  ドリフト     : {_f(pe.drift_rate_as_per_min, '+.2f', ' \"/min')}   "
          f"PEモデル寄与率: {_f((pe.explained_fraction or 0)*100, '.0f', '%')}   残差RMS: {_f(pe.residual_rms, '.2f', ' \"')}")
        others = [pk for pk in pe.peaks if pk.label == "非ウォーム成分"][:3]
        for pk in others:
            w(f"  その他の周期成分: {pk.period:.1f} s ±{pk.amplitude:.2f}\" (S/N {pk.snr:.0f})")
    if pe.message != "OK":
        w(f"  ※ {pe.message}")
    w("")
    w("■ Dec バックラッシュ")
    if bl.ok:
        w(f"  推定値 : {bl.backlash_ms:.0f} ms (1σ {bl.backlash_ms_lo:.0f}–{bl.backlash_ms_hi:.0f} ms)"
          f"  ≈ {bl.backlash_arcsec:.1f}\"   信頼度: {bl.confidence}")
        w(f"  反転回数: {bl.n_reversals}   ゲイン比(フィット/キャリブ): {_f(bl.gain_ratio, '.2f')}"
          f"   PHD2 補正設定: {_f(bl.phd2_comp_ms, '.0f', ' ms')}")
    if bl.message != "OK":
        w(f"  ※ {bl.message}")
    w("")
    w("■ 振動・外乱")
    w(f"  ナイキスト周波数 {vib.nyquist_hz:.3f} Hz（これより速い機械振動は露出内で平均化され直接は分離不可）")
    w(f"  ノイズ床 (seeing+重心誤差): RA {vib.noise_ra:.2f}\"  Dec {vib.noise_dec:.2f}\"")
    w(f"  閉ループ lag-1 自己相関: RA {vib.closed_lag1_ra:+.2f}  Dec {vib.closed_lag1_dec:+.2f}   "
      f"パルス方向反転率: RA {_f(vib.alt_frac_ra*100, '.0f', '%')}  Dec {_f(vib.alt_frac_dec*100, '.0f', '%')}")
    w(f"  過補正振動: RA {'あり' if vib.oscillation_ra else 'なし'} / Dec {'あり' if vib.oscillation_dec else 'なし'}")
    nonworm = [pk for pk in vib.peaks if not pk["worm_harmonic"]][:4]
    for pk in nonworm:
        w(f"  短周期成分 {pk['axis']:>3}: {pk['period']:.1f} s ±{pk['amplitude']:.2f}\" (S/N {pk['snr']:.0f})")
    w(f"  突発外乱: {len(vib.bursts)} 件 (時間比 {vib.burst_time_fraction*100:.1f}%)")
    for b in vib.bursts[:8]:
        w(f"    t={b.t_start:7.1f}–{b.t_end:7.1f} s  最大 {b.peak_arcsec:.1f}\"  {b.axes:6s} → {b.likely}")
    w("")
    w("■ 予測制御シミュレーション (RA)")
    if ev.get("ok"):
        w(f"  {ev['message']}")
        w(f"  フィードバックのみ RMS {ev['baseline_rms']:.2f}\"  →  +PEC予測 RMS {ev['pec_rms']:.2f}\""
          f"  (改善 {ev['improvement']*100:+.0f}%)")
        if a.forecast:
            nz = [c for c in a.forecast if c["pulse_ms"] > 0]
            w(f"  次 {len(a.forecast)} ステップのフィードフォワード指令を生成（例: "
              + ", ".join(f"{c['pulse_ms']:.0f}ms {c['direction']}" for c in nz[:4]) + " …）")
            w("  ※ 指令は生成のみで赤道儀には送信していません")
    else:
        w(f"  {ev.get('message')}")
    w("")
    w("■ 推奨事項")
    for r in recommendations(a):
        w(f"  ・{r}")
    return "\n".join(L)


def plot_report(a: Analysis, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]

    pe, bl, vib, ev = a.pe, a.backlash, a.vibration, a.pec_eval
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    t_min = a.openloop.t / 60

    # 1 RA drift + PE model
    A = ax[0, 0]
    A.plot(t_min, a.openloop.ra_raw_as, lw=0.6, color="0.6", label="閉ループ誤差 (RA)")
    if pe.ok:
        trend = pe.drift_stitched - pe.pe_model - pe.residual
        A.plot(t_min, pe.drift_stitched, lw=0.7, color="C0", label="再構成した補正前ドリフト")
        A.plot(t_min, trend + pe.pe_model, lw=1.5, color="C3", label="PE + トレンドモデル")
    A.set(title="RA トラッキング誤差", xlabel="時間 [min]", ylabel="arcsec")
    A.legend(fontsize=8)

    # 2 periodogram
    A = ax[0, 1]
    if pe.freqs is not None:
        A.semilogx(1 / pe.freqs, pe.spectrum, lw=0.8)
        if pe.worm_period:
            for n in range(1, pe.n_harmonics + 1):
                A.axvline(pe.worm_period / n, color="C3", ls=":", lw=0.8)
    A.set(title="RA ドリフト振幅スペクトル (赤点線=ウォーム高調波)", xlabel="周期 [s]", ylabel="振幅 [arcsec]")

    # 3 folded PE
    A = ax[1, 0]
    if pe.ok:
        ph = ((pe.t - pe.t[0]) % pe.worm_period) / pe.worm_period
        A.scatter(ph, pe.pe_model + pe.residual, s=2, alpha=0.3, color="C0")
        o = np.argsort(ph)
        A.plot(ph[o], pe.pe_model[o], color="C3", lw=2)
        A.set(title=f"位相折り畳み PE (周期 {pe.worm_period:.1f} s, P-P {pe.pp_sky:.1f}\")",
              xlabel="ウォーム位相", ylabel="arcsec")

    # 4 backlash profile
    A = ax[1, 1]
    if bl.ok:
        A.plot(bl.grid, bl.rss / bl.rss.min(), color="C2")
        A.axvline(bl.backlash_ms, color="C3")
        A.axvspan(bl.backlash_ms_lo, bl.backlash_ms_hi, color="C3", alpha=0.15)
        A.set(title=f"Dec バックラッシュ推定 {bl.backlash_ms:.0f} ms ({bl.backlash_arcsec:.1f}\")",
              xlabel="バックラッシュ幅 [ms]", ylabel="相対残差平方和")
    else:
        A.text(0.5, 0.5, bl.message, ha="center", va="center", wrap=True, transform=A.transAxes)
        A.set_title("Dec バックラッシュ")

    # 5 residuals + bursts
    A = ax[2, 0]
    if vib.res_ra is not None:
        A.plot(t_min, vib.res_ra, lw=0.5, label="RA 残差")
        A.plot(t_min, vib.res_dec, lw=0.5, label="Dec 残差")
        for b in vib.bursts:
            A.axvspan(b.t_start / 60, b.t_end / 60, color="red", alpha=0.2)
    A.set(title="高周波残差と突発外乱 (赤帯)", xlabel="時間 [min]", ylabel="arcsec")
    A.legend(fontsize=8)

    # 6 PEC replay
    A = ax[2, 1]
    if ev.get("ok"):
        tt = ev["t"] / 60
        A.plot(tt, ev["baseline_error"], lw=0.6, label=f"フィードバックのみ RMS {ev['baseline_rms']:.2f}\"")
        A.plot(tt, ev["pec_error"], lw=0.6, label=f"+PEC予測 RMS {ev['pec_rms']:.2f}\"")
        A.legend(fontsize=8)
    A.set(title="予測制御シミュレーション (RA)", xlabel="時間 [min]", ylabel="arcsec")

    fig.suptitle(f"AstroSense Guiding — {a.session.start}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
