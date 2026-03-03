from pathlib import Path

from scripts.phase56_autonomous_engine import build_graph, calculate_snr, validate_graph


def test_phase56_graph_is_valid():
    graph = build_graph(Path("."))
    validate_graph(graph)


def test_phase56_graph_weights_sum_to_one():
    graph = build_graph(Path("."))
    total = sum(node.weight for node in graph.values() if node.command is not None)
    assert total == 1.0


def test_phase56_snr_calculation():
    graph = build_graph(Path("."))
    for node in graph.values():
        if node.command is not None:
            node.status = "passed"

    signal, noise, snr = calculate_snr(graph)
    assert signal == 1.0
    assert noise == 0.0
    assert snr == 1.0
