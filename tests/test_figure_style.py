"""Unit tests for the figures._style visual system."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from figures._style import (
    ABLATION,
    CLASSICAL,
    DANN,
    DATASET_COLOURS,
    DOUBLE_COL,
    PROPOSED,
    PROPOSED_HATCH,
    SINGLE_COL,
    WORK1,
    WORK2,
    add_significance,
    apply_style,
    load_results,
    save_fig,
)


def test_colour_constants_are_unique():
    """Assert all primary method colour constants are non-empty and unique."""
    colours = [PROPOSED, WORK1, WORK2, ABLATION, DANN, CLASSICAL]
    assert len(colours) == len(set(colours)), "Colour constants must be distinct"
    for c in colours:
        assert c.startswith("#")
        assert len(c) == 7

    assert PROPOSED_HATCH == "//"


def test_dataset_colours_are_unique():
    """Assert dataset colours cover all 6 datasets with unique colours."""
    expected_datasets = {"nsl_kdd", "unsw_nb15", "ton_iot", "bot_iot", "cicids2017", "apa_ddos"}
    assert set(DATASET_COLOURS.keys()) == expected_datasets
    assert len(DATASET_COLOURS.values()) == len(set(DATASET_COLOURS.values()))


def test_widths_and_style_rcparams():
    """Assert dimension constants and apply_style rcParams configuration."""
    assert SINGLE_COL == 3.5
    assert DOUBLE_COL == 7.16

    family = apply_style()
    assert family in ("Arial", "DejaVu Sans")

    import matplotlib as mpl
    assert mpl.rcParams["figure.dpi"] == 300
    assert mpl.rcParams["axes.titlesize"] == 12
    assert mpl.rcParams["axes.labelsize"] == 10
    assert mpl.rcParams["xtick.labelsize"] == 9
    assert mpl.rcParams["ytick.labelsize"] == 9
    assert mpl.rcParams["legend.fontsize"] == 9
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["axes.grid"] is True
    assert mpl.rcParams["grid.color"] == "#DDDDDD"
    assert mpl.rcParams["grid.linewidth"] == 0.5
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["ps.fonttype"] == 42


def test_save_fig_emits_png_and_pdf(tmp_path: Path):
    """Assert save_fig writes both .png at 300 DPI and .pdf vector, and closes the figure."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    ax.plot([0, 1, 2], [10, 20, 30], color=PROPOSED)

    out_dir = tmp_path / "figs"
    paths = save_fig(fig, "test_plot", out_dir=out_dir)

    assert len(paths) == 2
    png_path = out_dir / "test_plot.png"
    pdf_path = out_dir / "test_plot.pdf"

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0

    # Ensure figure is closed
    assert not plt.fignum_exists(fig.number)


def test_add_significance():
    """Assert add_significance brackets place correct labels for p-values."""
    fig, ax = plt.subplots()
    add_significance(ax, 0.0, 1.0, 10.0, p_value=0.0005) # ***
    add_significance(ax, 1.0, 2.0, 10.0, p_value=0.005)  # **
    add_significance(ax, 2.0, 3.0, 10.0, p_value=0.03)   # *
    add_significance(ax, 3.0, 4.0, 10.0, p_value=0.10)   # ns

    texts = [t.get_text() for t in ax.texts]
    assert texts == ["***", "**", "*", "ns"]
    plt.close(fig)


def test_load_results(tmp_path: Path):
    """Assert load_results reads JSON dictionary and handles missing files cleanly."""
    fake_json = tmp_path / "results.json"
    fake_data = {"test_metric": 0.95}
    fake_json.write_text(json.dumps(fake_data))

    # Test load_results
    data = load_results()
    assert isinstance(data, dict)

    loaded = load_results(fake_json)
    assert loaded == fake_data

    # Missing file returns empty dict without throwing
    missing = load_results(tmp_path / "non_existent.json")
    assert missing == {}


def test_fig01_pipeline_generates(tmp_path):
    from figures.fig01_pipeline import draw_pipeline_figure
    out = draw_pipeline_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig01_pipeline.png").exists()
    assert (tmp_path / "fig01_pipeline.pdf").exists()


def test_fig02_domain_adversarial_generates(tmp_path):
    from figures.fig02_domain_adversarial import draw_domain_adversarial_figure
    out = draw_domain_adversarial_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig02_domain_adversarial.png").exists()
    assert (tmp_path / "fig02_domain_adversarial.pdf").exists()


def test_fig03_graph_fusion_generates(tmp_path):
    from figures.fig03_graph_fusion import draw_graph_fusion_figure
    out = draw_graph_fusion_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig03_graph_fusion.png").exists()
    assert (tmp_path / "fig03_graph_fusion.pdf").exists()


def test_fig04_same_domain_generates(tmp_path):
    from figures.fig04_same_domain import draw_same_domain_figure
    out = draw_same_domain_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig04_same_domain.png").exists()
    assert (tmp_path / "fig04_same_domain.pdf").exists()


def test_fig05_zero_shot_generates(tmp_path):
    from figures.fig05_zero_shot import draw_zero_shot_figure
    out = draw_zero_shot_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig05_zero_shot.png").exists()
    assert (tmp_path / "fig05_zero_shot.pdf").exists()


def test_fig06_domain_discriminator_generates(tmp_path):
    from figures.fig06_domain_discriminator import draw_domain_discriminator_figure
    out = draw_domain_discriminator_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig06_domain_discriminator.png").exists()
    assert (tmp_path / "fig06_domain_discriminator.pdf").exists()


def test_fig07_robustness_generates(tmp_path):
    from figures.fig07_robustness import draw_robustness_figure
    out = draw_robustness_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig07_robustness.png").exists()
    assert (tmp_path / "fig07_robustness.pdf").exists()


def test_fig08_convergence_generates(tmp_path):
    from figures.fig08_convergence import draw_convergence_figure
    out = draw_convergence_figure(out_dir=tmp_path)
    assert len(out) == 4
    assert (tmp_path / "fig08_convergence.png").exists()
    assert (tmp_path / "fig08_convergence.pdf").exists()
    assert (tmp_path / "fig08_supp_topologies.png").exists()
    assert (tmp_path / "fig08_supp_topologies.pdf").exists()


def test_fig09_mitigation_generates(tmp_path):
    from figures.fig09_mitigation import draw_mitigation_figure
    out = draw_mitigation_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig09_mitigation.png").exists()
    assert (tmp_path / "fig09_mitigation.pdf").exists()


def test_fig10_edge_generates(tmp_path):
    from figures.fig10_edge import draw_edge_figure
    out = draw_edge_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig10_edge.png").exists()
    assert (tmp_path / "fig10_edge.pdf").exists()


def test_fig11_zero_delay_generates(tmp_path):
    from figures.fig11_zero_delay import draw_zero_delay_figure
    out = draw_zero_delay_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig11_zero_delay.png").exists()
    assert (tmp_path / "fig11_zero_delay.pdf").exists()


def test_fig12_shap_summary_generates(tmp_path):
    from figures.fig12_shap_summary import draw_shap_summary_figure
    out = draw_shap_summary_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig12_shap_summary.png").exists()
    assert (tmp_path / "fig12_shap_summary.pdf").exists()


def test_fig13_attention_rollout_generates(tmp_path):
    from figures.fig13_attention_rollout import draw_attention_rollout_figure
    out = draw_attention_rollout_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig13_attention_rollout.png").exists()
    assert (tmp_path / "fig13_attention_rollout.pdf").exists()


def test_fig14_alert_report_generates(tmp_path):
    from figures.fig14_alert_report import draw_alert_report_figure
    out = draw_alert_report_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig14_alert_report.png").exists()
    assert (tmp_path / "fig14_alert_report.pdf").exists()


def test_fig15_shap_ig_correlation_generates(tmp_path):
    from figures.fig15_shap_ig_correlation import draw_shap_ig_correlation_figure
    out = draw_shap_ig_correlation_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig15_shap_ig_correlation.png").exists()
    assert (tmp_path / "fig15_shap_ig_correlation.pdf").exists()


def test_fig16_ablation_waterfall_generates(tmp_path):
    from figures.fig16_ablation_waterfall import draw_ablation_waterfall_figure
    out = draw_ablation_waterfall_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig16_ablation_waterfall.png").exists()
    assert (tmp_path / "fig16_ablation_waterfall.pdf").exists()


def test_fig17_robustness_utility_generates(tmp_path):
    from figures.fig17_robustness_utility import draw_robustness_utility_figure
    out = draw_robustness_utility_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig17_robustness_utility.png").exists()
    assert (tmp_path / "fig17_robustness_utility.pdf").exists()


def test_fig18_pareto_generates(tmp_path):
    from figures.fig18_pareto import draw_pareto_figure
    out = draw_pareto_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig18_pareto.png").exists()
    assert (tmp_path / "fig18_pareto.pdf").exists()


def test_fig19_radar_generates(tmp_path):
    from figures.fig19_radar import draw_radar_figure
    out = draw_radar_figure(out_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / "fig19_radar.png").exists()
    assert (tmp_path / "fig19_radar.pdf").exists()


