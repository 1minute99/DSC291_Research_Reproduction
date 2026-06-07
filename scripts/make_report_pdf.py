"""Assemble all figures / results / visualizations into one explained PDF.

One page per figure: section label, title, the figure, and a clear explanation
(what it is, how to read it, and why it matters). Numbers are the 10% overlay
run (the canonical reproduction). Reproducible — reads results/figs/.

    python scripts/make_report_pdf.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figs"
SLIDE = FIG / "slides"
PAPER = FIG / "paper"
OUT = ROOT / "MILAN - Figures and Results.pdf"

A4 = (8.27, 11.69)
INK = "#1b2430"
ACCENT = "#2E3B4E"
GREY = "#5b6b7b"

# (section, title, image path, [explanation paragraphs])
ENTRIES = [
    ("METHOD", "The MILAN editing pipeline", SLIDE / "milan_pipeline_diagram.png", [
        "MILAN describes a neuron by feeding its top-activating image patches (its \"exemplars\") into a "
        "pretrained, image-conditioned language model trained on 60k human neuron descriptions (the "
        "MILANNOTATIONS dataset). The output is a free-form English caption per neuron — e.g. \"words and "
        "letters\" or \"dog faces\" — rather than a label drawn from a fixed concept list, which is exactly "
        "what sets MILAN apart from earlier interpretability methods.",
        "Section 7 of the paper turns those captions into the editing loop shown above: train a classifier on "
        "spurious data, caption every neuron, flag the text-selective ones with a simple keyword filter, zero "
        "(ablate) them, and re-measure adversarial accuracy.",
        "No retraining or gradient surgery is involved. Because the captions are human-readable, the description "
        "itself tells you which neurons to remove — that is what makes an otherwise opaque network editable "
        "\"by name\", and it is the capability the rest of this document tests.",
    ]),
    ("DATASET", "Spurious-text Imagenette (how the shortcut is built)", SLIDE / "spurious_dataset_grid.png", [
        "Top row: clean Imagenette images (10 ImageNet classes), with no overlay.",
        "Middle row: the training set. A fraction (10%) of images have their CORRECT class name painted in the "
        "top-left corner. Because that text is perfectly predictive of the label, a model is tempted to read it "
        "instead of looking at the object — a spurious shortcut planted on purpose.",
        "Bottom row: the adversarial test set, where every image carries a WRONG class name. This is the crux of "
        "the design: a model that truly recognizes objects is unaffected, but a model that learned to read the "
        "corner text is now actively misled, so its accuracy collapses. The size of that collapse is precisely "
        "the robustness gap the editing experiment sets out to repair.",
    ]),
    ("REFERENCE — THE PAPER", "Paper Fig 7: MILAN names the text neuron", PAPER / "paper_fig7.png", [
        "From Hernandez et al. (2022). Panels (a) and (b) show training examples with the corner text and the "
        "adversarial test set with wrong-class text.",
        "Panel (c) shows the top exemplars of a single neuron (layer3-134) that MILAN captions \"words and "
        "letters\" — the spurious feature, named in plain English.",
        "This is the qualitative claim we set out to reproduce: rather than staying hidden, the shortcut is "
        "surfaced by MILAN as one human-readable phrase. That phrase is what makes the downstream editing "
        "possible — you can search captions for \"text\" and immediately have a list of neurons to remove.",
    ]),
    ("REFERENCE — THE PAPER", "Paper Fig 8: ablation recovers robustness", PAPER / "paper_fig8.png", [
        "The paper's headline editing result. The x-axis is the number of neurons zeroed; the y-axis is "
        "adversarial accuracy.",
        "The blue \"sort text\" curve ablates the neurons MILAN captioned as text and climbs from ~58% to ~63%, "
        "clearly beating the orange \"sort all\" baseline that removes neurons by overall importance. The dashed "
        "line marks the no-text-distractor ceiling.",
        "The important point is not just that accuracy rises, but that the caption-guided ordering beats an "
        "importance-based one: deleting the MILAN-named neurons restores most of the lost robustness with no "
        "retraining. This is the curve we reproduce on our own data over the next three pages.",
    ]),
    ("OUR REPRODUCTION (10% OVERLAY)", "Results at a glance", SLIDE / "summary_metrics_table.png", [
        "A summary of the reproduction plus the three extensions, all on the 10% overlay setting.",
        "Baseline ResNet18: 75.4% clean accuracy, 51.5% adversarial accuracy, and MILAN flags 216/1024 (21.1%) "
        "of neurons as text-selective. Ablating those neurons recovers adversarial accuracy and beats both the "
        "random and importance baselines, so the paper's core editing claim reproduces.",
        "The three extension rows probe how far that result generalizes — to a second architecture (VGG16), "
        "across network depth, and to a vision-language model (CLIP). Each is detailed on the following pages, "
        "and they all point in a consistent direction.",
    ]),
    ("OUR REPRODUCTION (10% OVERLAY)", "Our Fig 7: the text neurons we found", FIG / "fig7.png", [
        "Eight MILAN-flagged text neurons from our ResNet18. Each row is one neuron, and the row label on the "
        "left is MILAN's caption for it.",
        "The painted class words visible inside the exemplar thumbnails (\"pump\", \"truck\", \"springer\", …) "
        "are exactly the corner text these neurons respond to.",
        "That correspondence is the qualitative sanity check behind the whole method: the flagged units are "
        "genuinely reading text, not arbitrary neurons that happened to match the keyword filter. Because the "
        "captions are accurate here, they can be trusted as editing targets — which the next page puts to the "
        "test quantitatively.",
    ]),
    ("OUR REPRODUCTION (10% OVERLAY)", "Our Fig 8: the editing curve", FIG / "fig8.png", [
        "Our reproduction of the editing result. From the 51.5% no-ablation baseline (dotted), zeroing MILAN's "
        "text neurons (blue, \"text-sorted\") recovers adversarial accuracy to ~55% and holds it.",
        "Crucially, ablating the SAME number of neurons by importance (orange, \"sort-all\") or at random (grey, "
        "with the shaded ±1 std band) instead DEGRADES accuracy, to ~46% and ~37%. The green dashed line is "
        "clean accuracy (0.75).",
        "This is the key control in the whole project: it is the MILAN captions, not the mere act of removing "
        "neurons, that recover robustness — without them, the same number of deletions makes the model worse. "
        "Text-sorted is the only curve that stays above its starting line, so the captions are doing real work.",
    ]),
    ("ADDITIONAL EXPERIMENT 1", "Does editing generalize? (VGG16)", FIG / "fig_arch_comparison.png", [
        "Per-architecture ablation curves, with solid = text-sorted (MILAN-guided) and dashed = sort-all "
        "(importance-ordered).",
        "VGG16 is MORE shortcut-reliant than ResNet18 — its no-ablation adversarial accuracy is only 19.8% vs "
        "51.5%. Even so, ablating MILAN's text neurons recovers it strongly (19.8% → 32.7%, +12.8pp) and clearly "
        "beats its own importance baseline.",
        "We changed no code between the two architectures, only the trained model. So the editing recipe is "
        "model-agnostic across CNNs: wherever the shortcut happens to land, MILAN's captions still locate the "
        "neurons that carry it, and removing them helps.",
    ]),
    ("ADDITIONAL EXPERIMENT 1", "Text-neuron fraction across architectures", SLIDE / "arch_text_neuron_bar.png", [
        "The share of neurons MILAN labels as text-selective, per model. A higher bar means the model leans more "
        "heavily on the text shortcut.",
        "ResNet18 (21.1%, 216/1024) and VGG16 (24.3%, 357/1472) are task-trained CNNs and rely on the shortcut "
        "heavily. CLIP ViT-B/32 is only 7.9% (61/768) and is used zero-shot — it was never trained on the "
        "spurious set.",
        "The contrast previews the robustness result on the CLIP page: how much a model depends on this shortcut "
        "is governed less by the raw architecture than by how the model was pretrained.",
    ]),
    ("ADDITIONAL EXPERIMENT 2", "Where does the shortcut live? (layer depth)", FIG / "fig_layer_analysis.png", [
        "Panel (a): the fraction of text-selective neurons by ResNet18 layer jumps 3.0× from conv1 (14%) to "
        "layer1 (42%) — the shortcut appears right after the first residual block — then thins out with depth "
        "(layer2–4: 29%, 30%, 13%).",
        "Panel (b): caption diversity falls monotonically with depth, meaning deeper layers produce more "
        "repetitive, object-like descriptions.",
        "Together these say the text shortcut crystallizes very early in the network, and that depth mostly adds "
        "object structure on top rather than more text-reading features. That is also practically useful: the "
        "neurons worth editing out are concentrated in the early-to-mid layers.",
    ]),
    ("ADDITIONAL EXPERIMENT 3", "Is CLIP more robust? (zero-shot ViT-B/32)", FIG / "fig_clip_analysis.png", [
        "MILAN run on CLIP's vision-transformer blocks, with CLIP used zero-shot (no training on the spurious "
        "set). Panel (a) shows the text-neuron fraction by block, peaking at just 7.9% — about 2.7× lower than "
        "ResNet18 (21.1%).",
        "Panel (b) lists the top words per block: \"objects\", \"colored\", \"white\" — not \"words\"/\"letters\". "
        "CLIP simply does not devote many units to reading the overlay.",
        "So susceptibility to this shortcut is structural rather than incidental: CLIP's image-language "
        "pretraining yields representations far less reliant on the text overlay than a task-trained CNN — a "
        "concrete, measurable robustness advantage, and a fitting close to the generalization story.",
    ]),
]


def add_cover(pdf):
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.78, "MILAN — Editing Spurious Features", ha="center", fontsize=24,
             weight="bold", color=ACCENT)
    fig.text(0.5, 0.73, "Figures, Results & Visualizations", ha="center", fontsize=15, color=INK)
    fig.text(0.5, 0.685, "Reproduction of Hernandez et al., ICLR 2022  ·  10% overlay setting",
             ha="center", fontsize=11, color=GREY)
    intro = (
        "This document collects every figure from the project — the method, the dataset, the paper's own "
        "results, our reproduction, and the three additional experiments — and explains, for each one, what it "
        "shows, how to read it, and why it matters. All quantitative results use the 10% training-overlay "
        "setting, the configuration that gives the clearest editing signal."
    )
    fig.text(0.12, 0.625, textwrap.fill(intro, 92), fontsize=11, va="top", color=INK)
    fig.text(0.12, 0.49, "Contents", fontsize=13, weight="bold", color=ACCENT)
    y = 0.458
    last = None
    for entry in ENTRIES:
        sec, title = entry[0], entry[1]
        if sec != last:
            fig.text(0.12, y, sec, fontsize=9.5, weight="bold", color=GREY)
            y -= 0.0195
            last = sec
        fig.text(0.16, y, f"– {title}", fontsize=10, color=INK)
        y -= 0.0225
    fig.text(0.12, 0.035, "Team: Wonmin Kim · Seongho Kim · Ming-Yang Wu · Steven Tsai",
             fontsize=9, color=GREY)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def add_page(pdf, section, title, img_path, paragraphs):
    fig = plt.figure(figsize=A4)
    fig.text(0.07, 0.96, section, fontsize=10, weight="bold", color=GREY)
    fig.text(0.07, 0.935, title, fontsize=16, weight="bold", color=ACCENT, va="top")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.915, 0.915], color="#c9d2dc", lw=1))
    # figure in the upper area, letterboxed to preserve aspect
    ax = fig.add_axes([0.07, 0.45, 0.86, 0.44])
    if img_path.exists():
        ax.imshow(mpimg.imread(str(img_path)))
    else:
        ax.text(0.5, 0.5, f"missing: {img_path.name}", ha="center")
    ax.axis("off")
    # explanation
    body = "\n\n".join(textwrap.fill(p, 95) for p in paragraphs)
    fig.text(0.07, 0.41, body, fontsize=10.5, va="top", color=INK, linespacing=1.4)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def main():
    with PdfPages(str(OUT)) as pdf:
        add_cover(pdf)
        for sec, title, path, paras in ENTRIES:
            add_page(pdf, sec, title, path, paras)
    print(f"wrote {OUT}  ({len(ENTRIES) + 1} pages)")


if __name__ == "__main__":
    main()
