"""Build reports/report.docx. Run from /home/user/mlr-project:
    /home/user/.venv/bin/python reports/build_report.py
All numbers come from results/*.csv or reports/discussion.md (inserted verbatim)."""
import os
import re

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "reports", "report.docx")
rd = lambda f: pd.read_csv(os.path.join(RES, f))

doc = Document()
styles = doc.styles
TABLE_STYLE = "Light Grid Accent 1" if "Light Grid Accent 1" in [s.name for s in styles] else "Table Grid"


def set_font(style, size=None, bold=None, color=None):
    style.font.name = "Calibri"
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), "Calibri")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold


for nm in ("Normal", "List Bullet", "List Bullet 2", "Caption", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"):
    set_font(styles[nm])
styles["Normal"].font.size = Pt(11)
styles["Normal"].paragraph_format.space_after = Pt(6)
styles["Caption"].font.size = Pt(9.5)
styles["Heading 1"].font.size = Pt(16)
styles["Heading 2"].font.size = Pt(13)
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Inches(1)
    sec.top_margin = sec.bottom_margin = Inches(0.9)

counters = {"fig": 0, "tab": 0, "fig_n": 0}
# Table numbers are sequential by insertion order; this list fixes the order so in-text references resolve.
# table(..., key=...) asserts that the key lands on the slot listed here.
TAB_ORDER = ["dataset", "uci", "seg", "segacc", "temporal", "tempacc", "odo", "reg", "ac5", "blind", "clust", "perm",
             "rl", "ac8", "shield", "v2", "vis", "visacc", "vishard", "vishardacc", "suit"]
tref = lambda k: f"Table {TAB_ORDER.index(k) + 1}"


def add_runs(p, text, bold_all=False):
    """Inline markdown (**bold**) -> runs."""
    for i, part in enumerate(re.split(r"\*\*", text)):
        if part:
            r = p.add_run(part)
            r.bold = bold_all or (i % 2 == 1)
    return p


def para(text="", style=None, align=None):
    p = doc.add_paragraph(style=style)
    add_runs(p, text)
    if align is not None:
        p.alignment = align
    return p


def bullet(text, level=1):
    return para(text, "List Bullet" if level == 1 else "List Bullet 2")


def caption(kind, text, label_override=None):
    label = "Figure" if kind == "fig" else "Table"
    if label_override is None:
        counters[kind] += 1
        tag = f"{label} {counters[kind]}"
    else:  # un-numbered-sequence label, so cited Figure numbers elsewhere do not shift
        tag = f"{label} {label_override}"
    p = doc.add_paragraph(style="Caption")
    r = p.add_run(f"{tag}: ")
    r.bold = True
    p.add_run(text)
    return p


def figure(fname, text, label_override=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(RES, fname), width=Inches(6))
    c = caption("fig", text, label_override)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    counters["fig_n"] += 1


def table(header, rows, caption_text, size=9, widths=None, bold_rows=(), key=None):
    if key is not None:
        assert TAB_ORDER.index(key) + 1 == counters["tab"] + 1, (key, counters["tab"])
    cp = caption("tab", caption_text)
    cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = TABLE_STYLE
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(size)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            add_runs(cells[i].paragraphs[0], ("–" if str(v).strip().lower() in ("nan", "nan ± nan") else str(v)), bold_all=ri in bold_rows)
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(size)
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    for ri_, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
        if ri_ == 0:
            th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0)
    if widths:
        t.autofit = False
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


f3 = lambda x: f"{x:.3f}"
pm = lambda m, s: f"{m:.3f} ± {s:.3f}" if pd.notna(s) else f"{m:.3f}"
passfmt = lambda v: {True: "Yes", False: "No"}.get(v, "n/a") if not isinstance(v, str) else ("Yes" if v == "True" else "No" if v == "False" else "n/a")

# ======================================================================= TITLE PAGE
for _ in range(5):
    doc.add_paragraph()
p = doc.add_paragraph(style="Title")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Intelligent Obstacle Detection and Navigation Analysis for Autonomous Robots Using Machine Learning")
for line, b in [("Machine Learning for Robotics — Section 9A", True), ("Instructor: Mr. Sudais", False)]:
    q = para("", align=WD_ALIGN_PARAGRAPH.CENTER)
    r = q.add_run(line)
    r.font.size = Pt(14)
    r.bold = b
doc.add_paragraph()
q = para("", align=WD_ALIGN_PARAGRAPH.CENTER)
q.add_run("Group").bold = True
for m in ["Syeda Hamna Uraizee (22k-5112)", "Ayesha Wahab (22k-5087)", "Hamdan (22k-5077)"]:
    para(m, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
para("8 November 2026", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# ======================================================================= 1 INTRO
doc.add_heading("1 Introduction & Problem Statement", 1)
para("Autonomous mobile robots rely on range sensors such as LiDAR and ultrasonic transducers to perceive their surroundings. "
     "The raw readings are noisy, high-dimensional and partly incomplete (dropouts, limited fields of view), so turning them into "
     "reliable decisions is a machine-learning problem as much as an engineering one. This project studies four related problems "
     "on such data: (i) classifying what the sensors see, (ii) estimating the distance to the nearest obstacle, (iii) clustering "
     "scans into environment types, and (iv) learning navigation policies with reinforcement learning (RL). A further experiment adds a "
     "simulated camera to test whether appearance resolves the ambiguity left by 2D range data, alone and fused with LiDAR.")
para("These problems matter because errors in perception or control translate directly into collisions, inefficient paths and unsafe "
     "behaviour. A robot that mislabels an obstacle, underestimates a distance or follows an over-aggressive policy is a hazard to itself "
     "and to people around it. We therefore evaluate every model with leakage-safe protocols and metrics that match the task, and we "
     "report where the models fail as well as where they succeed.")

# ======================================================================= 2 OBJECTIVES
doc.add_heading("2 Objectives", 1)
for b in ["Engineer informative features from raw LiDAR, ultrasonic and wall-following sensor data.",
          "Build and compare classification, regression and clustering models on these features.",
          "Apply reinforcement learning to robot navigation and compare it with a reactive baseline.",
          "Analyse the strengths and weaknesses of each model family.",
          "Use evaluation metrics that are appropriate to each task.",
          "Recommend suitable approaches for different robotics applications."]:
    bullet(b)

# ======================================================================= 3 DATA
doc.add_heading("3 Data", 1)
doc.add_heading("3.1 Simulator", 2)
para("All supervised and RL experiments use a custom 2D simulator (sim/). The arena is 8 × 8 m and is built by one of three map "
     "generators: open (a few boxes and cylinders), narrow (serpentine corridors formed by alternating wall segments) and cluttered "
     "(many obstacle objects). Each map also contains two small cylindrical targets. The map type is a deterministic function of the map seed.")
para("The robot carries two sensor suites. A 360-ray LiDAR (maximum range 5 m) is used at full resolution for object segmentation and "
     "downsampled to 36 rays for frame features and RL. Its noise is Gaussian with σ = 1% · r + 1 cm, with 2% of rays dropped out. "
     "Eight ultrasonic sensors with 30° cones are spaced 45° apart; each reading is the minimum over the cone, with 3% multiplicative "
     "noise and 4% dropout. Because cones are 30° wide and spaced 45°, there are 15° gaps between cones.")
para("For the vision experiment (Section 6.5) a simulated pan camera is aimed at each LiDAR segment and renders a 32 × 32 RGB crop "
     "(camera height 0.3 m, 40° vertical field of view, ±3° horizontal padding around the segment, minimum horizontal field of view 8°). "
     "Appearance is assigned per object: walls are grey/beige and 1.0 m tall; targets are red–orange, 0.4–0.6 m tall, with a horizontal stripe 60% of the time; "
     "obstacles have a uniformly random hue and are 0.3–1.2 m tall. Rendering noise comprises distance shading, a global brightness factor of 0.6–1.4, "
     "a ±5% per-channel tint and Gaussian pixel noise with σ = 8/255. This appearance model was authored by the team, and the designed class overlap is small.")
para("Data are collected by a controller that follows the reactive rule (Section 6.4) but takes a uniformly random action with probability "
     "ε = 0.2. Each of the 300 maps (100 per type) is driven for 3 episodes of 150 steps; a frame is recorded every 5 steps and a "
     "360-ray scan is segmented every 10 steps. Table 1 summarises the resulting dataset.")
t1 = rd("T1_dataset.csv")
names = {"maps": "Maps", "episodes": "Episodes", "frames": "Frames", "segments_total": "Segments (total, before filtering)",
         "segments_dropped_small(<3pts)": "Segments dropped (fewer than 3 points)",
         "segments_dropped_impure(<90%)": "Segments dropped (label purity below 90%)", "segments_kept": "Segments kept",
         "frames_observable_frac": "Fraction of frames whose nearest point lies inside an ultrasonic cone"}
rows = [[names.get(r["item"], r["item"]), f"{r['value']:,.0f}" if r["value"] > 1 else f3(r["value"])] for _, r in t1.iterrows()]
table(["Item", "Value"], rows, "Simulated dataset summary (results/T1_dataset.csv).", size=10, widths=[4.6, 1.4], key="dataset")

doc.add_heading("3.2 UCI Wall-Following Robot Navigation data", 2)
para("As a real-sensor benchmark we use the UCI Wall-Following Robot Navigation dataset in its 4-sensor version (front, left, right and back "
     "distances; 5,456 rows), recorded on a SCITOS G5 robot by Freire et al. (2009) and licensed CC BY 4.0. The four classes are "
     "Move-Forward, Slight-Right-Turn, Sharp-Right-Turn and Slight-Left-Turn. The class distribution is imbalanced, so macro-F1 is used.")

# ======================================================================= 4 PROTOCOL
doc.add_heading("4 Evaluation protocol", 1)
for b in ["**Grouped cross-validation.** Simulator tasks use GroupKFold(5) grouped by map_seed so that no map appears in both training and test folds. Only out-of-fold predictions are reported.",
          "**No leakage through preprocessing.** Imputation, scaling and polynomial expansion sit inside scikit-learn Pipelines and are fitted on training rows only.",
          "**Tuning.** Hyper-parameters are chosen by an inner GroupKFold(3) grid search on the training part of each outer fold, with at most 6 configurations per model.",
          "**Pre-specified primary model.** Random Forest was designated the primary model before running experiments; all acceptance criteria use it.",
          "**Uncertainty.** Confidence intervals are obtained by bootstrapping over maps (not rows).",
          "**Disjoint seed ranges.** Supervised data use map seeds 100000–100299, RL training uses 0–9999 and RL evaluation uses 200000–200149 (50 maps per type); disjointness is asserted in code.",
          "**UCI data.** Rows are time-ordered, so 5-fold evaluation uses contiguous blocks with a 50-row gap between the test block and the training rows."]:
    bullet(b)
para("Five acceptance criteria were fixed in advance:")
a2 = rd("T2_acceptance.csv").rename(columns={"pass": "ok"})
a3 = rd("T3_acceptance.csv").rename(columns={"pass": "ok"})
a5 = rd("T5_acceptance.csv").rename(columns={"pass": "ok"})
bullet("**AC1 (segment classification):** " + a2.criterion[0].replace("AC1a: ", "(a) ") + "; and " + a2.criterion[1].replace("AC1b: ", "(b) ") + ". AC1 passes only if both hold.")
bullet("**AC5 (distance regression):** " + a3.criterion[0].replace("AC5a: ", "(a) ") + "; and " + a3.criterion[2].replace("AC5b: ", "(b) ") + ". AC5 passes only if both hold.")
bullet("**AC8 (RL navigation):** PPO success of at least 80% on open maps and at least 60% on narrow and on cluttered maps (held-out evaluation maps), and a PPO collision rate no higher than the reactive controller in each map type.")
bullet("**AC2 (vision):** CNN pooled out-of-fold macro-F1 of at least 0.80.")
bullet("**AC3 (fusion):** late-fusion macro-F1 minus LiDAR Random Forest macro-F1 of at least +0.02, with the lower bound of the bootstrap 95% CI (resampling maps) above 0.")
para("Post-hoc analyses (the RL safety shield and temporal tracking for segment classification) were added after the main results were seen; "
     "they are reported separately from the pre-specified acceptance criteria and are not counted towards them. Three further analyses were pre-specified as follow-ups "
     "(their design was fixed before any of their results was computed) but were run after the main results, so they too are exploratory and are not counted towards the acceptance criteria: "
     "robustness of temporal tracking to odometry noise (Section 6.1), an ablation of PPO with a stronger collision penalty and frame stacking (Section 6.4), and a hard-appearance variant of the camera experiment (Section 6.5).")
para("Reproducibility: all seeds are fixed. A full rerun from a fresh clone (data generation, all supervised tasks, vision and RL "
     "evaluation; about 44 minutes on one cloud machine) reproduced every result table exactly, apart from wall-clock timings. "
     "PPO and Q-learning training was not rerun; evaluation uses the saved models.")
para("The vision task uses the same 60,000-segment subsample and the same GroupKFold(5)-by-map outer folds as segment classification. Within each outer-training set, "
     "an inner 80/20 split by maps provides the validation data for CNN early stopping and fusion-weight selection.")

# ======================================================================= 5 FEATURES
doc.add_heading("5 Feature engineering", 1)
bullet("**Segment geometry (LiDAR).** Each LiDAR segment (consecutive valid hits with range jumps of at most 0.15 m) is described by point count, angular width, mean/min/std range, chord length, path length, path-to-chord ratio, linearity residual (smallest singular value), circle-fit radius and residual, a bulge indicator and the end-to-end range difference. The raw baseline uses the point count, mean range and 8 resampled ranges.")
bullet("**Ultrasonic aggregates.** Minimum, mean and standard deviation over the 8 readings; front, side and back minima; number of dropouts; number of readings below 1 m.")
bullet("**Clustering window features.** For windows of 10 consecutive frames: mean range per LiDAR ray, mean and standard deviation of the per-frame minimum, free-ray fraction, front/left/right/back sector minima and the mean number of range jumps.")
bullet("**UCI features.** The four raw distances plus minimum, maximum, mean, standard deviation, front–left difference, front/right ratio, left–right difference and causal lag-1 differences of each sensor.")

# ======================================================================= 6 RESULTS
doc.add_heading("6 Results", 1)
doc.add_heading("6.1 Classification", 2)
para("UCI wall-following. Table 2 gives 5-fold blocked-CV results for all six models with raw and engineered features; "
     "Figure 1 plots macro-F1 and Figure 2 shows the Random Forest confusion matrix.")
u = rd("uci_summary.csv")
order = ["Majority", "LogReg", "KNN", "RandomForest", "XGBoost", "MLP"]
u["o"] = u.model.map(order.index)
u = u.sort_values(["o", "features"])
rows = [[r.model, r.features, pm(r.macro_f1_mean, r.macro_f1_std), pm(r.accuracy_mean, r.accuracy_std)] for r in u.itertuples()]
table(["Model", "Features", "Macro-F1 (mean ± std)", "Accuracy (mean ± std)"], rows,
      "UCI wall-following classification, blocked 5-fold CV (results/uci_summary.csv).", widths=[1.4, 1.2, 1.8, 1.8], key="uci")
figure("uci_f1_bar.png", "UCI macro-F1 by model and feature set.")
figure("uci_confusion_rf.png", "Random Forest confusion matrix on UCI (out-of-fold).")

para("Simulated LiDAR segments (wall / obstacle / target). Segment results use a uniform 60,000-row subsample of the 212,825 kept segments, "
     "with folds grouped by map. Table 3 compares models, Table 4 reports the acceptance criteria and Figure 3 shows the confusion matrix.")
s = rd("T2_segment_classification.csv")
s["o"] = s.model.map(["Dummy", "LogReg", "KNN", "RandomForest", "XGBoost", "MLP"].index)
s = s.sort_values(["o", "features"])
rows = [[r.model + (" (primary)" if r.primary else ""), r.features, pm(r.macro_f1_mean, r.macro_f1_std), pm(r.accuracy_mean, r.accuracy_std),
         f3(r.recall_wall), f3(r.recall_obstacle), f3(r.recall_target)] for r in s.itertuples()]
table(["Model", "Features", "Macro-F1", "Accuracy", "Recall wall", "Recall obst.", "Recall target"], rows,
      "Segment classification on the 60,000-row subsample, GroupKFold(5) by map (results/T2_segment_classification.csv).", size=8.5,
      widths=[1.3, 0.9, 1.1, 1.1, 0.6, 0.6, 0.6], key="seg")


def acc_rows(df, with_ci=True):
    out = []
    for r in df.itertuples():
        v = "" if pd.isna(r.value) else f3(r.value)
        ci = f"[{f3(r.ci_low)}, {f3(r.ci_high)}]" if pd.notna(r.ci_low) else ""
        out.append([r.criterion, v, ci, passfmt(r.ok)])
    return out


table(["Criterion", "Value", "95% CI", "Pass"], acc_rows(a2), "Segment classification acceptance criteria and audit rows (results/T2_acceptance.csv).",
      size=8.5, widths=[3.6, 0.7, 1.2, 0.5], key="segacc")
figure("F2_segment_confusion.png", "Segment classification confusion matrix (Random Forest, engineered features).")

para("**Exploratory: temporal tracking.** This analysis was added after the main results and is post-hoc; it is not one of the pre-specified acceptance criteria. "
     "It asks whether several views of the same object resolve the ambiguity left by a single scan. Each segment is linked causally to the nearest segment within 0.3 m "
     "in the previous scan of the same episode (10 steps earlier), using a greedy one-to-one assignment on pose and centroid only; labels are never used for linking. "
     "Features are aggregated over up to the last 5 views: mean and max of point count, width, chord, circle radius, linearity and circle residuals and bulge; "
     "standard deviation of chord and radius; mean range; viewing-angle spread; track length; and the number of views in the window. The same 60,000-segment subsample, the same folds and the same Random Forest grid are used. "
     "Table 5 gives the results and Table 6 the AC1 re-check, the gain over single-scan features with a map-bootstrap CI, and descriptive gains by track length "
     "(Figure 3a plots macro-F1 by track length).")
tt = rd("T2c_temporal.csv")
rows = [[r.model, r.features, pm(r.macro_f1_mean, r.macro_f1_std), f3(r.pooled_macro_f1), f3(r.accuracy_mean),
         f3(r.recall_wall), f3(r.recall_obstacle), f3(r.recall_target)] for r in tt.itertuples()]
table(["Model", "Features", "Macro-F1 (mean ± std)", "Pooled macro-F1", "Accuracy", "Recall wall", "Recall obst.", "Recall target"], rows,
      "Segment classification with temporal tracking features, 60,000 segments, GroupKFold(5) by map (results/T2c_temporal.csv).", size=8.5,
      widths=[0.9, 1.1, 1.1, 0.7, 0.7, 0.5, 0.5, 0.5], key="temporal")
a2c = rd("T2c_acceptance.csv").rename(columns={"pass": "ok"})
table(["Criterion", "Value", "95% CI", "Pass"], acc_rows(a2c),
      "Post-hoc temporal-tracking checks: AC1 re-check, gain with CI, and descriptive gains by track length (results/T2c_acceptance.csv).",
      size=8.5, widths=[3.6, 0.7, 1.2, 0.5], key="tempacc")
figure("F10_temporal_gain.png", "Macro-F1 by track length: engineered vs engineered + temporal features (Random Forest).", label_override="3a")
para("These results rest on favourable assumptions. Segment poses are the exact simulator poses (perfect odometry); links are always correct because objects are spaced farther apart "
     "than the 0.3 m link radius, so a link can never cross to a different object; and only purity-filtered segments are used. The gain is therefore optimistic, "
     "and real scenes with denser objects and odometry drift would add tracking errors. The first assumption is relaxed in the odometry-noise analysis below; the other two are not.")
para("**Exploratory: odometry-noise robustness.** This is an exploratory analysis that was pre-specified but run after the main results. It replaces the exact poses with a dead-reckoning estimate. "
     "The first scan of each episode is anchored at the true pose; every later scan-to-scan increment (10 steps) is the true increment, expressed in the previous robot frame, plus Gaussian noise "
     "in x and y (σ_t) and in heading (σ_h), composed onto the previous estimated pose, so the pose error is a random walk that accumulates within the episode. "
     "The noise is per scan interval and is not proportional to the distance travelled (a stationary robot still drifts). Four levels were fixed in advance: none, low (σ_t = 0.02 m, σ_h = 1°), "
     "medium (0.05 m, 3°) and high (0.10 m, 6°). Segment linking and the viewing-angle spread use the estimated poses (segment shape features are in the robot frame and are unchanged); labels are never used. "
     "The Random Forest is fixed at the parameters selected in the T2 and T2c inner grids (min_samples_leaf = 5, max_features = sqrt), with the same 60,000 segments and folds; the engineered-only reference uses the same fixed parameters. "
     f"{tref('odo')} gives the link rate (fraction of segments linked to an earlier segment), pooled macro-F1 with a map-bootstrap 95% CI (1000 resamples), the gain over single-scan engineered features with its CI, and the median pose drift "
     "(position error in metres, heading error in degrees, over all scans). Figure 3b plots the same results.")
od = rd("T2d_odometry.csv")
od_ref = od[od.features == "engineered"].iloc[0]
od = od[od.features == "engineered+temporal"]
lv = {"none": "none", "low": "low (0.02 m, 1°)", "medium": "medium (0.05 m, 3°)", "high": "high (0.10 m, 6°)"}
ci = lambda m, lo, hi: f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
rows = [[lv[r.level], f"{r.link_rate:.3f}", ci(r.pooled_macro_f1, r.pooled_ci_low, r.pooled_ci_high),
         "+" + ci(r.gain_vs_eng, r.gain_ci_low, r.gain_ci_high), f"{r.median_pos_err_m:.3f} m / {r.median_head_err_deg:.1f}°"] for r in od.itertuples()]
table(["Noise level (σ_t, σ_h per scan)", "Link rate", "Pooled macro-F1 [95% CI]", "Gain vs single scan [95% CI]", "Median drift (position / heading)"], rows,
      f"Temporal tracking under odometry noise, Random Forest with fixed parameters (results/T2d_odometry.csv); single-scan engineered reference: pooled macro-F1 {od_ref.pooled_macro_f1:.3f}.",
      size=8.5, widths=[1.6, 0.7, 1.5, 1.5, 1.2], key="odo")
figure("F11_odometry.png", "Temporal-tracking macro-F1 and gain under dead-reckoning odometry noise (noise per scan interval).", label_override="3b")
_o = lambda lv_, c: od.loc[od.level == lv_, c].iloc[0]
para(f"The gain degrades gradually rather than collapsing: it is +{_o('none','gain_vs_eng'):.3f} with exact poses and +{_o('high','gain_vs_eng'):.3f} (CI {_o('high','gain_ci_low'):.3f}–{_o('high','gain_ci_high'):.3f}) at the high level, "
     f"where the link rate falls from {_o('none','link_rate'):.3f} to {_o('high','link_rate'):.3f} and median position drift reaches {_o('high','median_pos_err_m'):.2f} m. The CI lower bound stays above zero at every level. "
     "The no-noise row reproduces the earlier temporal result, as it should. The analysis still uses the favourable object spacing and purity-filtered segments described above, and the noise is a simple random walk, so real odometry may behave differently.")

doc.add_heading("6.2 Regression", 2)
para("The regression task estimates the distance to the nearest obstacle from the eight ultrasonic readings only. The baseline is the raw "
     "minimum ultrasonic reading. Results are reported for all frames and for two strata: frames whose nearest point lies inside an ultrasonic "
     "cone, and frames whose nearest point lies between cones (the observable = False stratum). “Near-range” MAE is computed on frames with true distance below 0.5 m. "
     f"{tref('reg')} lists all models on all frames, plus the baseline and the Random Forest (primary) on each stratum.")
g = rd("T3_regression.csv")
sn = {"all": "all frames", "observable": "nearest point inside a cone", "blind_spot": "nearest point between cones"}
keep = g[(g.stratum == "all") | g.model.isin(["Baseline(min us)", "RandomForest"])]
rows, bold = [], []
for r in keep.itertuples():
    if r.primary:
        bold.append(len(rows))
    rows.append([r.model + (" (primary)" if r.primary else ""), r.features, sn[r.stratum], f3(r.MAE), f3(r.near_MAE), f3(r.R2)])
table(["Model", "Features", "Stratum", "MAE (m)", "Near-range MAE (m)", "R²"], rows,
      "Distance regression, out-of-fold (results/T3_regression.csv); primary model in bold.", size=8.5,
      widths=[1.4, 0.9, 1.7, 0.6, 0.8, 0.6], bold_rows=bold, key="reg")
para(f"Acceptance criteria for AC5 are in {tref('ac5')}.")
table(["Criterion", "Value", "95% CI", "Pass"], acc_rows(a3), "Regression acceptance criteria, AC5 (results/T3_acceptance.csv).", size=8.5,
      widths=[3.6, 0.7, 1.2, 0.5], key="ac5")
para(f"A post-hoc analysis ({tref('blind')}) isolates the “truly blind” frames, where no cone sees the nearest point, and compares the baseline with "
     "Random Forest on engineered features.")
b = rd("T3b_truly_blind.csv")
sl = {"truly_blind": "truly blind", "seen": "seen by a cone", "between_cones (observable=False)": "nearest point between cones", "all": "all frames"}
rows = [[sl.get(r.stratum, r.stratum), f"{r.n:,}", f3(r.baseline_mae), f3(r.rf_eng_mae), f3(r.baseline_near_mae), f3(r.rf_eng_near_mae)] for r in b.itertuples()]
table(["Stratum", "n", "Baseline MAE (m)", "RF-eng MAE (m)", "Baseline near MAE (m)", "RF-eng near MAE (m)"], rows,
      "Truly blind frames versus other strata (results/T3b_truly_blind.csv).", size=8.5, widths=[1.9, 0.6, 0.9, 0.9, 0.9, 0.9], key="blind")
figure("F3_regression_mae_by_bin.png", f"Regression MAE by true-distance bin for three strata: nearest point seen by a cone, nearest point between cones (still seen by a cone edge), and truly blind ({tref('blind')}). Random Forest on engineered features (300 trees, same GroupKFold-by-map OOF protocol) vs the min-ultrasonic baseline.")

doc.add_heading("6.3 Clustering", 2)
para("Windowed LiDAR statistics were clustered with K-Means and a Gaussian mixture model (GMM) and compared with the hidden map type using the "
     f"adjusted Rand index (ARI). {tref('clust')} gives internal and external metrics; the reported solution is k = 3 (K-Means) and the BIC-best GMM.")
c = rd("T4_clustering.csv")
rows = [[r.algorithm, r.k, f3(r.silhouette), f3(r.db), f3(r.ari), f3(r.nmi), "" if pd.isna(r.bic) else f"{r.bic:,.0f}",
         "" if pd.isna(r.selected) else str(r.selected).strip()] for r in c.itertuples()]
table(["Algorithm", "k", "Silhouette", "Davies–Bouldin", "ARI", "NMI", "BIC", "Note"], rows,
      "Clustering results (results/T4_clustering.csv).", size=8.5, widths=[0.8, 0.3, 0.8, 0.9, 0.6, 0.6, 0.8, 1.7], key="clust")
para(f"{tref('perm')} compares the observed K-Means ARI with a map-level permutation null and a bootstrap over maps.")
pr = rd("T4_permutation.csv").iloc[0]
rows = [["Observed ARI (K-Means, k=3)", f3(pr.observed_ari)], ["Null mean", f3(pr.null_mean)], ["Null 99th percentile", f3(pr.null_p99)],
        ["Null maximum", f3(pr.null_max)], ["Permutation p-value", f3(pr.p_value)], ["Number of permutations", f"{int(pr.n_perm)}"],
        ["Bootstrap 95% CI for ARI", f"[{f3(pr.boot_ci_low)}, {f3(pr.boot_ci_high)}]"], ["Number of bootstrap resamples", f"{int(pr.n_boot)}"]]
table(["Quantity", "Value"], rows, "Permutation test and bootstrap for K-Means ARI (results/T4_permutation.csv).", size=10, widths=[3.5, 1.8], key="perm")
figure("F4_cluster_pca.png", "Clusters in PCA space.")
figure("F5_cluster_profiles.png", "Cluster centroid profiles.")

doc.add_heading("6.4 Reinforcement learning", 2)
para("Three navigation agents (plus post-hoc and ablation variants of PPO) are compared on the held-out evaluation maps (50 per map type): a rule-based reactive controller, tabular "
     "Q-learning and PPO. A uniformly random policy is included as a reference.")
para("**Environment and reward.** The observation is the 36 LiDAR rays divided by 5 m (dropout mapped to 1.0), the goal distance divided by 10 and the sine and "
     "cosine of the goal angle (39 dimensions). There are three discrete actions: forward 0.1 m, rotate +15° and rotate −15°. The reward is")
q = para("", align=WD_ALIGN_PARAGRAPH.CENTER)
q.add_run("r = 10·(d_goal[t−1] − d_goal[t]) − 0.05 − 0.5·[min_ray < 0.30 m] + 100·[goal] − 100·[collision]").italic = True
para("An episode ends on reaching the goal (distance below 0.2 m), on collision, or after 400 steps (truncation).")
para("**Reactive rule.** Using the minimum range in the front sector (±30°), the left sector (30–90°) and the right sector (−30 to −90°): if the front minimum is below 0.5 m, "
     "turn left when the left minimum is at least the right minimum, otherwise turn right; else if the goal bearing exceeds 15° turn toward the goal; otherwise move forward.")
para("**Tabular Q-learning.** The state combines 3 sectors (front, left, right) × 3 distance bins (below 0.3 m, 0.3–1.0 m, above 1.0 m) × 8 goal-angle bins, giving 27 × 8 = 216 states. "
     "Learning rate α = 0.1, discount γ = 0.99, ε decays linearly from 1.0 to 0.05 over the first 70% of 5,000 training episodes.")
para("**PPO.** Stable-Baselines3 PPO with an MLP policy and default hyper-parameters is trained for 300,000 steps with 3 seeds. Observations and rewards are normalised with VecNormalize during training; "
     "at evaluation the normalisation statistics are frozen (no updates, raw reward) and the policy acts deterministically.")
para("**Post-hoc safety shield.** After seeing PPO's collisions on the held-out maps we added a run-time veto, evaluated as a fifth agent (PPO + safety shield). "
     "When PPO chooses “forward”, the shield checks all 36 LiDAR rays: if any ray lies in the corridor the robot would sweep (lateral offset below 0.20 m and forward distance below 0.35 m), "
     "the action is replaced by the reactive rule's turn toward the more open side. Dropped-out rays count as free, and PPO is not retrained. Because the rule was designed after inspecting "
     "PPO's collisions and reuses the same held-out evaluation maps (a second use of that set), it is post-hoc and descriptive; it is not part of the pre-specified AC8. "
     "The 3.3% of collisions that remain on narrow maps arise from thin wall ends that fall between the 10° LiDAR rays, so the 36-ray sensor cannot see them.")
para("**Pre-specified ablation: PPO-v2.** This is an exploratory analysis that was pre-specified (rl/train_v2.py was written before any v2 training) but run after the main results. "
     "It tests the hypothesis that PPO's narrow-corridor failure comes from (a) a reward that under-penalises contact and (b) a memoryless single-frame observation. "
     "Relative to PPO, the collision penalty is raised from −100 to −200 and the proximity penalty (minimum ray below 0.30 m) from −0.5 to −1.0 per step; the last 4 observations are stacked with VecFrameStack "
     "applied after VecNormalize (so normalisation statistics stay per 39-dimensional frame and the policy sees 156 inputs); and training runs for 1,000,000 steps per seed with 3 seeds (PPO used 300,000). "
     "Everything else (environment, training map pool, Stable-Baselines3 defaults, MLP policy, evaluation maps) is unchanged. The unchanged post-hoc shield is also applied to PPO-v2. "
     "The support rule was fixed in advance: the hypothesis counts as supported if mean narrow-map success of PPO-v2 is at least 0.15 (absolute) above PPO and the worst PPO-v2 seed beats the best PPO seed on narrow maps; "
     "as partially supported if only narrow collisions fall or success rises by less than that; and as not supported otherwise. PPO-v2 was evaluated once on the same held-out maps; no hyper-parameter was tuned on them.")
para(f"{tref('rl')} reports per-agent, per-map-type results (mean ± std over 3 training seeds for Q-learning, PPO and PPO-v2; the reactive and random baselines are single deterministic runs, so no std is shown). "
     f"Path ratio is measured on successful episodes only. {tref('ac8')} reports the six pre-specified AC8 criteria, {tref('shield')} the post-hoc shield rows and {tref('v2')} the pre-specified PPO-v2 ablation rows.")
r5 = rd("T5_rl.csv")
AGENT_ORDER = ["random", "reactive", "qlearning", "ppo", "ppo_shield", "ppo_v2", "ppo_v2_shield"]
r5["o"] = r5.agent.map(AGENT_ORDER.index)
r5["m"] = r5.map_type.map(["open", "narrow", "cluttered"].index)
r5 = r5.sort_values(["o", "m"])
an = {"random": "Random", "reactive": "Reactive", "qlearning": "Q-learning", "ppo": "PPO", "ppo_shield": "PPO + safety shield (post-hoc)",
      "ppo_v2": "PPO-v2 (ablation)", "ppo_v2_shield": "PPO-v2 + shield (ablation)"}
rows = [[an[r.agent], r.map_type, pm(r.success_mean, r.success_std), pm(r.collision_mean, r.collision_std), pm(r.path_ratio_mean, r.path_ratio_std),
         pm(r.min_clear_mean, r.min_clear_std), pm(r.near_frac_mean, r.near_frac_std)] for r in r5.itertuples()]
table(["Agent", "Map type", "Success", "Collision", "Path ratio", "Min clearance (m)", "Near-collision fraction"], rows,
      "RL evaluation on held-out maps (results/T5_rl.csv).", size=8, widths=[1.1, 0.7, 0.95, 0.95, 0.95, 0.95, 0.95], key="rl")
def _v3(v):
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return v
ph = a5.criterion.str.startswith("[post-hoc]")
rows = [[r.criterion, _v3(r.value), passfmt(r.ok)] for r in a5[~ph & ~a5.criterion.str.startswith("[")].itertuples()]
table(["Criterion", "Value", "Pass"], rows, "RL acceptance criteria, AC8 (results/T5_acceptance.csv).", size=9, widths=[3.8, 1.5, 0.7], key="ac8")
rows = [[r.criterion.replace("[post-hoc] ", ""), _v3(r.value), passfmt(r.ok)] for r in a5[ph].itertuples()]
table(["Criterion", "Value", "Pass"], rows, "Post-hoc: PPO + safety shield against the AC8 thresholds (not pre-specified).", size=9, widths=[3.8, 1.5, 0.7], key="shield")
ab = a5[a5.criterion.str.startswith("[ablation v2]")]
rows = [[r.criterion.replace("[ablation v2] ", ""), _v3(r.value), passfmt(r.ok)] for r in ab.itertuples()]
table(["Criterion", "Value", "Pass"], rows, "Pre-specified ablation: PPO-v2 against the AC8 thresholds (results/T5_acceptance.csv, rows prefixed [ablation v2]; the original AC8 rows are unchanged).",
      size=9, widths=[3.8, 1.5, 0.7], key="v2")
_r = lambda ag, mt, col: r5[(r5.agent == ag) & (r5.map_type == mt)][col].iloc[0]
para("**PPO-v2 result (negative).** The ablation did not improve narrow-map success, so the hypothesis is not supported by the pre-specified rule. "
     f"Mean narrow success was {_r('ppo_v2','narrow','success_mean'):.3f} for PPO-v2 against {_r('ppo','narrow','success_mean'):.3f} for PPO (the rule required at least +0.15) and {_r('ppo_shield','narrow','success_mean'):.3f} for PPO + shield; "
     f"narrow collisions were unchanged at {_r('ppo_v2','narrow','collision_mean'):.3f} (PPO {_r('ppo','narrow','collision_mean'):.3f}). The change also cost performance elsewhere: open-map success fell from {_r('ppo','open','success_mean'):.3f} to {_r('ppo_v2','open','success_mean'):.3f} "
     f"and cluttered-map success from {_r('ppo','cluttered','success_mean'):.3f} to {_r('ppo_v2','cluttered','success_mean'):.3f}, so PPO-v2 now misses the 60% cluttered threshold that PPO met, while its collision rates (open {_r('ppo_v2','open','collision_mean'):.3f}, cluttered {_r('ppo_v2','cluttered','collision_mean'):.3f}) "
     f"are no better. With the shield, PPO-v2 reaches narrow success {_r('ppo_v2_shield','narrow','success_mean'):.3f} and {_r('ppo_v2_shield','narrow','collision_mean'):.3f} narrow collisions, again no better than PPO + shield. "
     "A larger penalty and short-term memory are therefore not enough to fix narrow corridors in this setup; the learning curves (Figure 8) show that the longer training did not change the training-time success plateau.")
figure("F6_rl_success.png", "Evaluation success rate by agent (random, reactive, Q-learning, PPO, PPO + safety shield, PPO-v2 and PPO-v2 + shield) and map type.")
figure("F7_rl_learning.png", "RL learning curves during training: PPO (solid) and PPO-v2 (dashed) for each seed, and Q-learning.")

doc.add_heading("6.5 Vision and LiDAR–camera fusion", 2)
v = rd("T6_vision.csv")
para("This experiment asks whether the camera crops of Section 3.1 resolve the obstacle/target ambiguity that limits LiDAR geometry (Section 6.1). "
     "Three image-based classifiers (plus a majority-class baseline), the LiDAR Random Forest and their late fusion are evaluated on the same 60,000 segments and outer folds. The **colour-rule baseline** involves no fitting: "
     "a hand-written three-rule heuristic on the central pixels. **Colour-histogram + logistic regression** is a simple learned baseline. The **CNN** has 3 convolutional "
     "blocks (16/32/64 filters) and is trained with Adam (learning rate 1e-3), batch size 256, a class-weighted loss and flip/brightness augmentation for at most 8 epochs, "
     "with early stopping (patience 3) on validation macro-F1; input normalisation uses training maps only. The **LiDAR Random Forest** uses the min_samples_leaf = 5 selected in the "
     "segment-classification task (Section 6.1) and is fitted on the same inner-training maps. **Late fusion** averages the class probabilities, p = w·p_CNN + (1 − w)·p_RF, "
     "with w chosen on the validation maps of each fold; the selected weights over the five folds were "
     + ", ".join(str(x) for x in v.loc[v.model == "Fusion", "fusion_w_per_fold"].iloc[0].split(";")) + ".")
para(f"{tref('vis')} gives the results and {tref('visacc')} the acceptance criteria.")
rows = [[r.model, pm(r.macro_f1_mean, r.macro_f1_std), f3(r.macro_f1_pooled), f3(r.accuracy), f3(r.recall_wall), f3(r.recall_obstacle), f3(r.recall_target)]
        for r in v.itertuples()]
table(["Model", "Macro-F1 (mean ± std over folds)", "Pooled macro-F1", "Accuracy", "Recall wall", "Recall obst.", "Recall target"], rows,
      "Vision and fusion results, GroupKFold(5) by map, 60,000 segments (results/T6_vision.csv).", size=8.5, widths=[1.5, 1.3, 0.8, 0.7, 0.6, 0.6, 0.6], key="vis")
a6 = rd("T6_acceptance.csv").rename(columns={"pass": "ok"})
note6 = a6[a6.criterion.str.startswith("note:")]
a6 = a6[~a6.criterion.str.startswith("note:")]
table(["Criterion", "Value", "95% CI", "Pass"], acc_rows(a6), "Vision acceptance criteria, AC2 and AC3 (results/T6_acceptance.csv).", size=8.5, widths=[3.6, 0.7, 1.2, 0.5], key="visacc")
for r in note6.itertuples():
    para("Note: " + r.criterion[len("note:"):].strip() + ".")
figure("F9_crop_examples.png", "Example simulated camera crops per class.")
figure("F8_vision_confusion.png", "Row-normalised confusion matrices: LiDAR RF, CNN, fusion.")
_g = lambda m, col: v.loc[v.model == m, col].iloc[0]
para(f"On synthetic crops the CNN reaches macro-F1 {f3(_g('CNN', 'macro_f1_pooled'))} and late fusion {f3(_g('Fusion', 'macro_f1_pooled'))}, "
     f"against {f3(_g('LiDAR-RF', 'macro_f1_pooled'))} for the LiDAR Random Forest on the same folds. The gap is expected: we designed the appearance model so that colour and a stripe "
     f"almost fully disambiguate the classes (a hand-written three-rule colour heuristic already reaches {f3(_g('ColourRule (hand, 3 rules)', 'macro_f1_pooled'))}), whereas 2D geometry cannot. "
     "The result shows that the pipeline, fusion and protocol work, not that a camera would deliver this margin in practice.")

para("**Hard-appearance variant.** This is an exploratory analysis that was pre-specified (sim.world.hard_appearance was fixed before any hard-mode result was computed) but run after the main results. "
     "Because the designed overlap in the main experiment is small, the variant makes appearance ambiguous. Walls: 85% grey/beige as before, 15% a saturated random colour. "
     "Obstacles: 30% take a target-like colour and a target-like height (0.4–0.6 m), the other 70% keep the original distribution; independently 20% carry a stripe. "
     "Targets: height 0.4–0.6 m; 80% target-like colour and 20% a random non-target hue; only 40% carry a stripe. "
     "In addition, 25% of crops are partially occluded: a vertical band covering 20–45% of the crop width at a uniformly random position, drawn as an unstriped object of random colour and height 0.3–1.2 m, "
     "at 0.5–0.9 times the depth of the nearest part of the segment. The rendering noise is unchanged. Geometry, LiDAR data, the 60,000 segments, folds, models, fusion protocol and criteria definitions are the same as in the easy experiment; "
     "the acceptance thresholds were set for the easy variant, so the hard-variant criteria are reported as descriptive only (pass column NA).")
vh = rd("T6b_vision_hard.csv")
para(f"{tref('vishard')} gives the results in the same format as {tref('vis')} and {tref('vishardacc')} the acceptance criteria and descriptive differences; Figure 11 shows example crops and Figure 12 the confusion matrices.")
rows = [[r.model, pm(r.macro_f1_mean, r.macro_f1_std), f3(r.macro_f1_pooled), f3(r.accuracy), f3(r.recall_wall), f3(r.recall_obstacle), f3(r.recall_target)]
        for r in vh.itertuples()]
table(["Model", "Macro-F1 (mean ± std over folds)", "Pooled macro-F1", "Accuracy", "Recall wall", "Recall obst.", "Recall target"], rows,
      "Vision and fusion results, hard-appearance variant, GroupKFold(5) by map, 60,000 segments (results/T6b_vision_hard.csv).", size=8.5, widths=[1.5, 1.3, 0.8, 0.7, 0.6, 0.6, 0.6], key="vishard")
cr = vh[vh.model.str.startswith("ColourRule")].iloc[0]
para("Note: the colour-rule baseline involves no fitting and its three rules were designed against the original (easy) appearance specification; it is unchanged and was not tuned for the hard variant.")
a6b = rd("T6b_acceptance_hard.csv")
note6b = a6b[a6b.criterion.str.contains("note:")]
a6b = a6b[~a6b.criterion.str.contains("note:")]
rows = []
for r in a6b.itertuples():
    ci_ = f"[{f3(r.ci_low)}, {f3(r.ci_high)}]" if pd.notna(r.ci_low) else ""
    rows.append([r.criterion.replace("[hard appearance] ", ""), "" if pd.isna(r.value) else f3(r.value), ci_, str(r._5)])
table(["Criterion", "Value", "95% CI", "Pass"], rows, "Vision acceptance criteria and descriptive differences, hard-appearance variant (results/T6b_acceptance_hard.csv).",
      size=8.5, widths=[3.4, 0.7, 1.2, 0.9], key="vishardacc")
for r in note6b.itertuples():
    para("Note: " + r.criterion.replace("[hard appearance] ", "").replace("note:", "", 1).strip() + ".")
figure("F13_crops_hard.png", "Example simulated camera crops per class, hard-appearance variant (including occluded crops).")
figure("F12_vision_hard_confusion.png", "Row-normalised confusion matrices, hard-appearance variant: LiDAR RF, CNN, fusion.")
_h = lambda m, col: vh.loc[vh.model == m, col].iloc[0]
_gr = a6b.set_index("criterion")
fc = _gr.loc["[hard appearance] descriptive: Fusion minus CNN macro-F1"]
fe = _g("Fusion", "macro_f1_pooled") - _g("CNN", "macro_f1_pooled")
para(f"On the hard variant the CNN falls to macro-F1 {f3(_h('CNN','macro_f1_pooled'))} (AC2 threshold 0.80 was set for the easy variant) and late fusion reaches {f3(_h('Fusion','macro_f1_pooled'))}, against {f3(_h('LiDAR-RF','macro_f1_pooled'))} for the unchanged LiDAR Random Forest. "
     f"Fusion adds +{fc.value:.3f} over the CNN (95% CI {fc.ci_low:.3f}–{fc.ci_high:.3f}), compared with +{fe:.3f} on easy appearance, so the two sensors are complementary when appearance is ambiguous. "
     "Target recall is the weak point for every model, and this remains a result on a team-authored appearance model.")

# ======================================================================= DISCUSSION (verbatim)
md = open(os.path.join(ROOT, "reports", "discussion.md"), encoding="utf-8").read().splitlines()
sections, cur = {}, None
for ln in md:
    if ln.startswith("## "):
        cur = ln[3:].strip()
        sections[cur] = []
    elif cur is not None:
        sections[cur].append(ln)


def render_md(lines):
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.lstrip().startswith("|"):
            i += 1  # tables handled separately
        elif re.match(r"^\s*-\s", ln):
            lvl = 1 + (len(ln) - len(ln.lstrip())) // 2
            bullet(re.sub(r"^\s*-\s+", "", ln), min(lvl, 2) if lvl <= 2 else 2)
            i += 1
        else:
            para(ln.strip())
            i += 1


def md_table(lines):
    tl = [l for l in lines if l.strip().startswith("|")]
    cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
    hdr = cells(tl[0])
    body = [cells(l) for l in tl[2:]]
    return hdr, body


doc.add_heading("7 Discussion", 1)
render_md(sections["Cross-task findings"])
doc.add_heading("8 Model suitability", 1)
suit_key = [k for k in sections if k.startswith("Model-suitability")][0]
pre = [l for l in sections[suit_key] if not l.strip().startswith("|")]
render_md(pre)
hdr, body = md_table(sections[suit_key])
table(hdr, body, "Model-suitability recommendations (from the discussion).", size=8.5, widths=[1.3, 1.6, 1.8, 1.8], key="suit")
doc.add_heading("9 Limitations & Future Work", 1)
render_md(sections["Limitations"])

doc.add_heading("10 Conclusion", 1)
_c = lambda ag, mt: r5[(r5.agent == ag) & (r5.map_type == mt)].collision_mean.iloc[0]
_s = lambda ag, mt: r5[(r5.agent == ag) & (r5.map_type == mt)].success_mean.iloc[0]
para("Across the tasks, tree ensembles were the most reliable default: Random Forest, the pre-specified primary model, was best or close to best on every supervised task, "
     "and feature engineering helped most where the raw signal was ambiguous. The acceptance criteria for segment classification (AC1) and distance regression (AC5) were not met, "
     "and the RL criterion (AC8) was only partially met, but each failure was informative: segment classification is limited by what a single 2D scan can observe, "
     "the raw minimum ultrasonic reading is already an excellent distance estimator and learning pays off mainly on truly blind frames, and PPO learned an aggressive goal-seeking policy that "
     "succeeds in open and cluttered maps but is unsafe and fails in narrow corridors, where the reactive rule or a planner remains safer. Clustering recovered the environment families, "
     "though only on synthetic families by design. "
     f"Two post-hoc follow-ups tested the suggested remedies: a run-time safety shield lowered PPO's collision rate from {_c('ppo','open'):.0%} to {_c('ppo_shield','open'):.0%} in open maps and from {_c('ppo','cluttered'):.0%} to {_c('ppo_shield','cluttered'):.0%} in cluttered maps "
     f"(success {_s('ppo_shield','open'):.3f} and {_s('ppo_shield','cluttered'):.3f}) but left {_c('ppo_shield','narrow'):.1%} collisions on narrow maps, and causal temporal tracking raised segment macro-F1 from {tt.pooled_macro_f1.iloc[0]:.3f} to {tt.pooled_macro_f1.iloc[1]:.3f} without reaching the AC1 threshold. "
     "Both are exploratory and optimistic by construction (the shield reuses the held-out maps; tracking uses exact poses). "
     f"Three pre-specified follow-ups, also exploratory, completed the picture: the tracking gain survives odometry noise, falling from +{_o('none','gain_vs_eng'):.3f} to +{_o('high','gain_vs_eng'):.3f} (CI lower bound {_o('high','gain_ci_low'):.3f}) at the highest noise level; "
     f"the PPO-v2 ablation did not help, with narrow-map success {_r('ppo_v2','narrow','success_mean'):.3f} against {_r('ppo','narrow','success_mean'):.3f} for PPO, so the hypothesis was not supported; "
     f"and under hard camera appearance the CNN fell to {f3(_h('CNN','macro_f1_pooled'))} while late fusion, at {f3(_h('Fusion','macro_f1_pooled'))}, added +{fc.value:.3f} over the CNN (CI {fc.ci_low:.3f}–{fc.ci_high:.3f}), showing that camera and LiDAR are complementary when appearance is ambiguous. "
     "Model choice should therefore follow the application, with a reactive safety layer around learned navigation, and real-camera validation with learned fusion, "
     "curriculum learning and real-sensor validation, listed above, are the natural next work.")

# ======================================================================= REFERENCES
doc.add_heading("References", 1)
for ref in ["Freire, A. L., Barreto, G. A., Veloso, M., & Varela, A. T. (2009). Short-term memory mechanisms in neural network learning of robot navigation tasks: A case study. 6th Latin American Robotics Symposium (LARS).",
            "UCI Machine Learning Repository: Wall-Following Robot Navigation Data (CC BY 4.0).",
            "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12.",
            "Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research, 22(268).",
            "Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
            "Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8."]:
    para(ref)

doc.save(OUT)

# ======================================================================= SUMMARY
d = Document(OUT)
print(f"saved {OUT}")
print(f"paragraphs={len(d.paragraphs)} tables={len(d.tables)} figures={counters['fig_n']} table_style={TABLE_STYLE}")
for p in d.paragraphs:
    if p.style.name.startswith("Heading"):
        print(("  " if p.style.name == "Heading 2" else "") + p.text)
