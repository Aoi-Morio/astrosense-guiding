# AstroSense Guiding オンライン予測評価

乱数seed: [1, 2, 3, 4, 5]。ウォーム周期 638.3 s、全12周期、学習待機3周期。
RLS忘却係数 0.995、4高調波。
観測間隔 2.15 s、補正レート 0.007 秒角/ms。
FB設定: {'aggression': 0.7, 'min_move_as': 0.3, 'hysteresis': 0.1, 'max_pulse_ms': 2500.0}。NumPy 2.5.4。

指標は真の追尾誤差RMS（秒角）。seeingを含む観測RMSと各実行値はJSON参照。
平均は同じseedを使った実行間平均。負の改善率は悪化を表す。

| 条件 | 評価区間 | FB | 固定PEC | RLS | RLS対FB | RLS対固定PEC |
|---|---|---:|---:|---:|---:|---:|
| stationary_low | 学習待機後すべて | 0.2966 | 0.1373 | 0.1480 | +50.1% | -7.8% |
| stationary_low | 8周期以降 | 0.2930 | 0.1358 | 0.1468 | +49.9% | -8.1% |
| stationary_typical | 学習待機後すべて | 0.4677 | 0.4209 | 0.4491 | +4.0% | -6.7% |
| stationary_typical | 8周期以降 | 0.4643 | 0.4206 | 0.4482 | +3.5% | -6.6% |
| stationary_high | 学習待機後すべて | 0.8640 | 0.8420 | 0.8992 | -4.1% | -6.8% |
| stationary_high | 8周期以降 | 0.8612 | 0.8409 | 0.8972 | -4.2% | -6.7% |
| amplitude_ramp | 学習待機後すべて | 0.5780 | 0.4609 | 0.4497 | +22.2% | +2.4% |
| amplitude_ramp | 8周期以降 | 0.5871 | 0.4641 | 0.4482 | +23.7% | +3.4% |
| amplitude_ramp_drops | 学習待機後すべて | 0.5959 | 0.4738 | 0.4563 | +23.4% | +3.7% |
| amplitude_ramp_drops | 8周期以降 | 0.6063 | 0.4791 | 0.4540 | +25.1% | +5.2% |

## 評価条件

- Known fixed period, cadence, and exact pulse calibration
- Instantaneous additive seeing measurement noise; no exposure integration
- Fixed PEC fits only the pre-warmup prefix; online updates only on current/past frames
- Amplitude doubles smoothly during the two cycles after warmup
- No gusts, dithers, actuator delay, calibration error, or changing worm period
- True tracking RMS includes dropped frames; measured RMS excludes them
- No parameter tuning or model selection on benchmark results

seeing条件は0.2/0.6/1.2秒角の独立な観測ノイズ。振幅変化条件では、
学習待機終了後の2周期でPEの全高調波の振幅が1倍から2倍へ線形に変わる。
欠測条件は5%のランダムな星喪失で、そのフレームでは学習と補正を止める。
固定PECは最初の3周期の観測を一括で最小二乗フィットし、その後は固定する。
RLSは逐次更新し、全方式とも同じフィードバック・パルス上限を使用する。

真の追尾誤差はシミュレータ側だけが採点に使う。制御器は閉ループ観測と自身の
補正履歴から学習する。未来の観測値や真値をモデル選択・指令生成に使わない。

合成データ・理想的な赤道儀モデルでの比較であり、実機性能は未検証。
既存ログ解析の0.83→0.80秒角は観測誤差の別評価で、この表とは直接比較しない。
