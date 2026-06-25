"""
KNN Predictions — Interactive CLI Inspector + Live Q&A
=======================================================
Usage:
    py test_predictions.py                  # interactive menu
    py test_predictions.py --row 5          # inspect a specific row
    py test_predictions.py --summary        # overall stats only
    py test_predictions.py --filter High    # show all High predictions
    py test_predictions.py --ask            # open Q&A assistant
"""

import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ── Colour codes ───────────────────────────────────────────────────────────────
R  = "\033[91m"
G  = "\033[92m"
Y  = "\033[93m"
B  = "\033[94m"
M  = "\033[95m"
C  = "\033[96m"
W  = "\033[1;97m"
DIM= "\033[2m"
RST= "\033[0m"

CLASS_COLOURS = {"Low": B, "Medium": G, "High": Y, "Very High": R}
PROB_COLS     = ["prob_low", "prob_medium", "prob_high", "prob_very_high"]
LABEL_TO_PROB = {"low":"prob_low","medium":"prob_medium",
                 "high":"prob_high","very high":"prob_very_high"}

def colour(text, code):  return f"{code}{text}{RST}"
def hdr(text):           print(colour(f"\n━━━  {text}  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", W))

def confidence_bar(prob, width=30):
    filled = int(prob * width)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = prob * 100
    col    = G if pct >= 70 else (Y if pct >= 40 else R)
    return f"{col}{bar}{RST} {pct:5.1f}%"

def confidence_label(prob):
    if prob >= 0.75: return colour("VERY HIGH ✔✔", G)
    if prob >= 0.55: return colour("HIGH      ✔",  G)
    if prob >= 0.40: return colour("MODERATE  ~",  Y)
    if prob >= 0.25: return colour("LOW       ✗",  R)
    return colour("VERY LOW  ✗✗", R)

def load_data(path="predictions.csv"):
    p = Path(__file__).resolve().parent / path
    if not p.exists():
        p = Path(path)
    if not p.exists():
        print(colour(f"\n  ✗  File not found: {path}", R))
        print(colour("     Place predictions.csv in the same folder as this script.\n", DIM))
        exit(1)
    return pd.read_csv(p)

# ══════════════════════════════════════════════════════════════════════════════
# LIVE Q&A  — answers computed directly from the predictions DataFrame
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLES = [
    "how many records are in the database",
    "how many phones are predicted as High",
    "what is the average confidence for Medium phones",
    "show me the row with the highest confidence",
    "how many predictions have confidence above 80%",
    "what percentage of phones are Very High",
    "which class has the most predictions",
    "which class has the lowest average confidence",
    "how many Low confidence predictions are there",
    "show the top 5 most confident High predictions",
    "what is the min and max confidence in the database",
    "how many rows have confidence below 50%",
    "compare average confidence across all classes",
    "how many phones are predicted Low or Medium",
    "what is the confidence distribution",
]

def show_examples():
    print(colour("\n  ── Sample questions about the database ───────────────", Y))
    for i, q in enumerate(EXAMPLES, 1):
        print(f"  {colour(str(i)+'.', DIM):<12} {q}")
    print()

def _max_prob(row):
    return row[PROB_COLS].max()

def _class_count(df, label):
    return (df["predicted_label"].str.lower() == label.lower()).sum()

def _class_prob_col(label):
    return LABEL_TO_PROB.get(label.lower().strip())

def answer_query(q, df):
    """
    Parse the natural language query and compute the answer live from df.
    Returns a list of answer lines, or None if not understood.
    """
    q_lower = q.lower().strip()
    lines   = []
    max_probs = df[PROB_COLS].max(axis=1)

    # ── total records ─────────────────────────────────────────────────────────
    if any(w in q_lower for w in ["how many records","total records","total rows",
                                   "how many rows","how many entries","size of","database size",
                                   "how many phones","total phones","how many data"]):
        lines += [
            f"Total records in the database : {colour(str(len(df)), W)}",
            f"Columns                       : {', '.join(df.columns.tolist())}",
        ]
        return lines

    # ── count by class ────────────────────────────────────────────────────────
    for cls in ["low","medium","high","very high"]:
        patterns = [f"predicted as {cls}", f"classified as {cls}",
                    f"in {cls}", f"are {cls}", f"how many {cls}",
                    f"count of {cls}", f"number of {cls}",f"{cls} predictions",
                    f"{cls} phones", f"{cls} class"]
        if any(p in q_lower for p in patterns):
            n   = _class_count(df, cls)
            pct = n / len(df) * 100
            col_name = _class_prob_col(cls)
            sub = df[df["predicted_label"].str.lower() == cls.lower()]
            avg_conf = sub[col_name].mean() * 100 if col_name and len(sub) else 0
            col = CLASS_COLOURS.get(cls.title(), W)
            lines += [
                f"Class           : {colour(cls.title(), col)}",
                f"Count           : {colour(str(n), W)}",
                f"Percentage      : {colour(f'{pct:.1f}%', W)}",
                f"Avg confidence  : {colour(f'{avg_conf:.1f}%', W)}",
            ]
            return lines

    # ── percentage / share of a class ─────────────────────────────────────────
    if "percentage" in q_lower or "percent" in q_lower or "share" in q_lower or "ratio" in q_lower:
        lines.append("Class distribution (% of total):")
        counts = df["predicted_label"].value_counts()
        for cls in ["Low","Medium","High","Very High"]:
            n   = counts.get(cls, 0)
            pct = n / len(df) * 100
            bar = colour("▓" * int(pct / 2), CLASS_COLOURS.get(cls, W))
            lines.append(f"  {colour(f'{cls:<10}', CLASS_COLOURS.get(cls,W))} {bar} {n} ({pct:.1f}%)")
        return lines

    # ── which class has most / least predictions ──────────────────────────────
    if ("most predictions" in q_lower or "highest count" in q_lower
            or "most phones" in q_lower or "dominant class" in q_lower):
        top = df["predicted_label"].value_counts().idxmax()
        n   = df["predicted_label"].value_counts().max()
        col = CLASS_COLOURS.get(top, W)
        lines += [f"Class with most predictions : {colour(top, col)} ({n} phones)"]
        return lines

    if ("least predictions" in q_lower or "lowest count" in q_lower
            or "fewest phones" in q_lower or "least phones" in q_lower):
        bot = df["predicted_label"].value_counts().idxmin()
        n   = df["predicted_label"].value_counts().min()
        col = CLASS_COLOURS.get(bot, W)
        lines += [f"Class with fewest predictions : {colour(bot, col)} ({n} phones)"]
        return lines

    # ── average confidence (overall or by class) ──────────────────────────────
    if any(w in q_lower for w in ["average confidence","avg confidence","mean confidence",
                                   "average prob","mean prob"]):
        # check if a class is mentioned
        found = None
        for cls in ["low","medium","high","very high"]:
            if cls in q_lower:
                found = cls; break
        if found:
            col_name = _class_prob_col(found)
            sub = df[df["predicted_label"].str.lower() == found]
            if col_name and len(sub):
                avg = sub[col_name].mean() * 100
                col = CLASS_COLOURS.get(found.title(), W)
                lines += [f"Avg confidence for {colour(found.title(), col)} : {colour(f'{avg:.1f}%', W)}"]
            else:
                lines += [f"No records found for class: {found}"]
        else:
            lines.append("Average confidence by class:")
            for cls in ["Low","Medium","High","Very High"]:
                col_name = _class_prob_col(cls)
                sub = df[df["predicted_label"] == cls]
                if col_name and len(sub):
                    avg = sub[col_name].mean() * 100
                    lines.append(f"  {colour(f'{cls:<10}', CLASS_COLOURS.get(cls,W))} {avg:.1f}%")
        return lines

    # ── compare confidence across classes ─────────────────────────────────────
    if "compare" in q_lower and ("confidence" in q_lower or "class" in q_lower):
        lines.append("Confidence comparison across all classes:")
        lines.append(f"  {'Class':<12} {'Mean':>8} {'Min':>8} {'Max':>8} {'Count':>7}")
        lines.append("  " + "─"*46)
        for cls in ["Low","Medium","High","Very High"]:
            col_name = _class_prob_col(cls)
            sub = df[df["predicted_label"] == cls]
            if col_name and len(sub):
                mn  = sub[col_name].mean() * 100
                mi  = sub[col_name].min()  * 100
                mx  = sub[col_name].max()  * 100
                col = CLASS_COLOURS.get(cls, W)
                lines.append(f"  {colour(f'{cls:<12}', col)} {mn:>7.1f}% {mi:>7.1f}% {mx:>7.1f}% {len(sub):>7}")
        return lines

    # ── which class has lowest/highest avg confidence ─────────────────────────
    if "lowest" in q_lower and "confidence" in q_lower:
        avgs = {}
        for cls in ["Low","Medium","High","Very High"]:
            col_name = _class_prob_col(cls)
            sub = df[df["predicted_label"] == cls]
            if col_name and len(sub):
                avgs[cls] = sub[col_name].mean() * 100
        if avgs:
            worst = min(avgs, key=avgs.get)
            col = CLASS_COLOURS.get(worst, W)
            lines += [f"Lowest avg confidence : {colour(worst, col)} ({avgs[worst]:.1f}%)"]
        return lines

    if "highest" in q_lower and "confidence" in q_lower and "class" in q_lower:
        avgs = {}
        for cls in ["Low","Medium","High","Very High"]:
            col_name = _class_prob_col(cls)
            sub = df[df["predicted_label"] == cls]
            if col_name and len(sub):
                avgs[cls] = sub[col_name].mean() * 100
        if avgs:
            best = max(avgs, key=avgs.get)
            col = CLASS_COLOURS.get(best, W)
            lines += [f"Highest avg confidence : {colour(best, col)} ({avgs[best]:.1f}%)"]
        return lines

    # ── row with highest/lowest confidence ────────────────────────────────────
    if ("highest confidence" in q_lower or "most confident" in q_lower or
            "best prediction" in q_lower) and "class" not in q_lower:
        idx  = max_probs.idxmax()
        prob = max_probs[idx]
        lbl  = df.loc[idx, "predicted_label"]
        col  = CLASS_COLOURS.get(lbl, W)
        lines += [
            f"Row with highest confidence : row {colour(str(idx), W)}",
            f"  Predicted : {colour(lbl, col)}",
            f"  Confidence: {confidence_bar(prob, 20)} {confidence_label(prob)}",
        ]
        return lines

    if "lowest confidence" in q_lower or "least confident" in q_lower or "worst prediction" in q_lower:
        idx  = max_probs.idxmin()
        prob = max_probs[idx]
        lbl  = df.loc[idx, "predicted_label"]
        col  = CLASS_COLOURS.get(lbl, W)
        lines += [
            f"Row with lowest confidence : row {colour(str(idx), W)}",
            f"  Predicted : {colour(lbl, col)}",
            f"  Confidence: {confidence_bar(prob, 20)} {confidence_label(prob)}",
        ]
        return lines

    # ── min and max confidence ────────────────────────────────────────────────
    if ("min" in q_lower or "max" in q_lower or "minimum" in q_lower or "maximum" in q_lower or
            "range" in q_lower) and "confidence" in q_lower:
        mn_val = max_probs.min() * 100
        mx_val = max_probs.max() * 100
        mn_row = max_probs.idxmin()
        mx_row = max_probs.idxmax()
        lines += [
            f"Minimum confidence : {colour(f'{mn_val:.1f}%', R)} (row {mn_row})",
            f"Maximum confidence : {colour(f'{mx_val:.1f}%', G)} (row {mx_row})",
            f"Range              : {colour(f'{mx_val - mn_val:.1f}%', W)}",
        ]
        return lines

    # ── count above/below a threshold ─────────────────────────────────────────
    thresh_match = re.search(r"(\d+)\s*%", q_lower)
    if thresh_match:
        t = int(thresh_match.group(1)) / 100
        if any(w in q_lower for w in ["above","over","greater","more than","higher"]):
            n   = (max_probs >= t).sum()
            pct = n / len(df) * 100
            lines += [
                f"Predictions with confidence ≥ {int(t*100)}% : {colour(str(n), G)} ({pct:.1f}%)",
            ]
            return lines
        if any(w in q_lower for w in ["below","under","less than","lower","fewer"]):
            n   = (max_probs < t).sum()
            pct = n / len(df) * 100
            lines += [
                f"Predictions with confidence < {int(t*100)}% : {colour(str(n), R)} ({pct:.1f}%)",
            ]
            return lines

    # ── low confidence count ──────────────────────────────────────────────────
    if "low confidence" in q_lower and ("how many" in q_lower or "count" in q_lower or "number" in q_lower):
        n   = (max_probs < 0.40).sum()
        pct = n / len(df) * 100
        lines += [f"Low confidence predictions (<40%) : {colour(str(n), R)} ({pct:.1f}%)"]
        return lines

    # ── top N most/least confident for a class ────────────────────────────────
    top_n_match = re.search(r"top\s*(\d+)", q_lower)
    n_top = int(top_n_match.group(1)) if top_n_match else 5
    if "top" in q_lower and ("confident" in q_lower or "confidence" in q_lower):
        found_cls = None
        for cls in ["low","medium","high","very high"]:
            if cls in q_lower: found_cls = cls; break
        sub = df[df["predicted_label"].str.lower() == found_cls] if found_cls else df
        col_name = _class_prob_col(found_cls) if found_cls else None
        if col_name:
            sub = sub.nlargest(n_top, col_name)
            label_str = found_cls.title()
        else:
            sub = sub.loc[max_probs.nlargest(n_top).index]
            label_str = "all classes"
        col = CLASS_COLOURS.get(found_cls.title() if found_cls else "Low", W)
        lines.append(f"Top {n_top} most confident predictions ({colour(label_str, col)}):")
        lines.append(f"  {'Row':<6} {'Class':<12} {'Confidence'}")
        lines.append("  " + "─"*36)
        for idx, row in sub.iterrows():
            lbl  = row["predicted_label"]
            prob = max_probs[idx]
            lcol = CLASS_COLOURS.get(lbl, W)
            lines.append(f"  {colour(str(idx),''):<10} {colour(f'{lbl:<12}', lcol)} {prob*100:.1f}%")
        return lines

    # ── confidence distribution buckets ───────────────────────────────────────
    if "distribution" in q_lower and "confidence" in q_lower:
        buckets = [(0, 0.25,"<25%"), (0.25,0.40,"25–40%"),
                   (0.40,0.55,"40–55%"), (0.55,0.70,"55–70%"), (0.70,1.01,"≥70%")]
        lines.append("Confidence distribution:")
        for lo, hi, label in buckets:
            n   = ((max_probs >= lo) & (max_probs < hi)).sum()
            pct = n / len(df) * 100
            bar = colour("▓" * int(pct / 2), G if lo >= 0.70 else (Y if lo >= 0.40 else R))
            lines.append(f"  {label:<8} {bar} {n} ({pct:.1f}%)")
        return lines

    # ── combined class count (Low or Medium) ─────────────────────────────────
    cls_found = [cls for cls in ["low","medium","high","very high"] if cls in q_lower]
    if len(cls_found) >= 2 and any(w in q_lower for w in ["or","and","combined","total"]):
        total = sum(_class_count(df, c) for c in cls_found)
        pct   = total / len(df) * 100
        labels = " + ".join(c.title() for c in cls_found)
        lines += [f"Combined count ({labels}) : {colour(str(total), W)} ({pct:.1f}%)"]
        return lines

    return None

def print_answer(lines):
    print(colour("\n  ┌" + "─"*56 + "┐", C))
    for line in lines:
        print(colour("  │ ", C) + line)
    print(colour("  └" + "─"*56 + "┘", C))

def ask_mode(df):
    print(colour("\n╔══════════════════════════════════════════════════════╗", C))
    print(colour("║     Mobile Price DB — Q&A Assistant 🤖               ║", C))
    print(colour("║     Ask anything about your predictions database     ║", C))
    print(colour("╚══════════════════════════════════════════════════════╝", C))
    print(colour(f"\n  Database loaded: {colour(str(len(df)), W)} records | "
                 f"{colour(str(df.shape[1]), W)} columns", DIM))
    show_examples()
    print(colour("  Type a question, 'examples' to list samples, or 'q' to go back.\n", DIM))

    while True:
        try:
            query = input(colour("  Ask › ", W)).strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not query: continue
        if query.lower() in ("q", "quit", "back", "exit"):
            break
        if query.lower() in ("examples", "help", "?"):
            show_examples(); continue

        result = answer_query(query, df)
        if result:
            print_answer(result)
        else:
            print(colour("\n  I couldn't compute that from the database.", R))
            print(colour("  Try rephrasing, or type 'examples' for ideas.\n", DIM))

# ── Standard views ─────────────────────────────────────────────────────────────
def print_header():
    print(colour("\n╔══════════════════════════════════════════════════════╗", C))
    print(colour("║       KNN Mobile Price — Prediction Inspector        ║", C))
    print(colour("╚══════════════════════════════════════════════════════╝", C))

def show_summary(df):
    max_probs = df[PROB_COLS].max(axis=1)
    hdr("OVERALL SUMMARY")
    print(f"  Total predictions     : {colour(str(len(df)), W)}")
    print(f"  Avg confidence        : {colour(f'{max_probs.mean()*100:.1f}%', W)}")
    print(f"  High confidence (≥70%): {colour(str((max_probs>=0.70).sum()), G)}")
    print(f"  Low  confidence (<40%): {colour(str((max_probs<0.40).sum()), R)}")

    hdr("CLASS DISTRIBUTION")
    counts = df["predicted_label"].value_counts()
    for cls in ["Low","Medium","High","Very High"]:
        n   = counts.get(cls, 0)
        pct = n / len(df) * 100
        bar = "▓" * int(pct / 2)
        col = CLASS_COLOURS.get(cls, W)
        print(f"  {colour(f'{cls:<10}',col)} {bar:<25} {n:>4}  ({pct:.1f}%)")

    hdr("CONFIDENCE STATS BY CLASS")
    for cls in ["Low","Medium","High","Very High"]:
        col_name = _class_prob_col(cls)
        sub = df[df["predicted_label"] == cls]
        if not len(sub): continue
        col = CLASS_COLOURS.get(cls, W)
        print(f"  {colour(f'{cls:<10}', col)}  "
              f"mean={sub[col_name].mean()*100:.1f}%  "
              f"min={sub[col_name].min()*100:.1f}%  "
              f"max={sub[col_name].max()*100:.1f}%")

def show_row(df, idx):
    if idx < 0 or idx >= len(df):
        print(colour(f"\n  ✗  Row {idx} out of range (0–{len(df)-1})", R)); return
    row      = df.iloc[idx]
    label    = row["predicted_label"]
    cls      = int(row["predicted_class"])
    probs    = {c.replace("prob_","").replace("_"," ").title(): row[c] for c in PROB_COLS}
    top_prob = max(probs.values())
    col      = CLASS_COLOURS.get(label, W)
    hdr(f"ROW {idx}")
    print(f"  Predicted Class : {colour(label, col)}  (class {cls})")
    print(f"  Confidence      : {confidence_label(top_prob)}")
    print()
    print(colour("  Probability breakdown:", DIM))
    for name, p in probs.items():
        marker = colour(" ◄ predicted", col) if name == label else ""
        print(f"    {name:<12} {confidence_bar(p)}{marker}")

def show_filter(df, label):
    label = label.title()
    if label == "Very_High": label = "Very High"
    sub = df[df["predicted_label"] == label].copy()
    if sub.empty:
        print(colour(f"\n  No predictions found for class: {label}", R)); return
    prob_col = _class_prob_col(label)
    sub  = sub.sort_values(prob_col, ascending=False)
    col  = CLASS_COLOURS.get(label, W)
    hdr(f"{label.upper()} predictions ({len(sub)} rows)")
    print(colour(f"  {'Row':<6} {'Confidence':>12}  {'Bar'}", DIM))
    print(colour("  " + "─"*50, DIM))
    for i, (orig_idx, r) in enumerate(sub.iterrows()):
        p = r[prob_col]
        print(f"  {colour(str(orig_idx), W):<17} {colour(f'{p*100:.1f}%', col):<14} {confidence_bar(p, 20)}")
        if i == 19:
            print(colour(f"  ... and {len(sub)-20} more rows", DIM)); break

def interactive_menu(df):
    print_header()
    while True:
        print(colour("\n  ┌─ MENU ──────────────────────────────────────┐", C))
        print(colour("  │  1  Overall summary & class stats            │", C))
        print(colour("  │  2  Inspect a specific row                   │", C))
        print(colour("  │  3  Filter by predicted class                │", C))
        print(colour("  │  4  Top 10 most confident predictions        │", C))
        print(colour("  │  5  Top 10 least confident predictions       │", C))
        print(colour("  │  6  Ask anything about the database  🤖      │", C))
        print(colour("  │  q  Quit                                     │", C))
        print(colour("  └──────────────────────────────────────────────┘", C))

        choice = input(colour("  Choose › ", W)).strip().lower()

        if choice == "1":   show_summary(df)
        elif choice == "2":
            try:
                idx = int(input(colour(f"  Row number (0–{len(df)-1}) › ", W)).strip())
                show_row(df, idx)
            except ValueError:
                print(colour("  ✗  Enter a valid number", R))
        elif choice == "3":
            print("  Classes: Low | Medium | High | Very High")
            label = input(colour("  Class › ", W)).strip()
            show_filter(df, label)
        elif choice == "4":
            max_p = df[PROB_COLS].max(axis=1)
            hdr("TOP 10 MOST CONFIDENT")
            for i in max_p.nlargest(10).index: show_row(df, i)
        elif choice == "5":
            max_p = df[PROB_COLS].max(axis=1)
            hdr("TOP 10 LEAST CONFIDENT")
            for i in max_p.nsmallest(10).index: show_row(df, i)
        elif choice == "6":
            ask_mode(df)
        elif choice == "q":
            print(colour("\n  Bye! 👋\n", C)); break
        else:
            print(colour("  ✗  Invalid choice", R))

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    if sys.platform == "win32":
        os.system("color")

    parser = argparse.ArgumentParser(description="KNN Prediction Inspector")
    parser.add_argument("--row",     type=int,         help="Inspect a specific row")
    parser.add_argument("--summary", action="store_true", help="Show overall summary")
    parser.add_argument("--filter",  type=str,         help="Filter by class")
    parser.add_argument("--ask",     action="store_true", help="Open Q&A assistant")
    args = parser.parse_args()

    df = load_data("predictions.csv")
    print_header()

    if args.row is not None:     show_row(df, args.row)
    elif args.summary:           show_summary(df)
    elif args.filter:            show_filter(df, args.filter)
    elif args.ask:               ask_mode(df)
    else:                        interactive_menu(df)