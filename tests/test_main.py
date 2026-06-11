"""Minimal tests for giftpack core logic."""
import io
import pandas as pd
import pytest

from main import parse_csv, solve


def make_csv(rows):
    content = "item,kg,value\n"
    for item, kg, value in rows:
        content += f"{item},{kg},{value}\n"
    return io.BytesIO(content.encode())


def test_parse_csv_basic():
    f = make_csv([("apple", 0.5, 2.0), ("book", 1.0, 5.0)])
    df = parse_csv(f)
    assert len(df) == 2
    assert list(df.columns) == ["item", "kg", "value"]


def test_parse_csv_drops_invalid():
    f = make_csv([("apple", "bad", 2.0), ("book", 1.0, 5.0)])
    df = parse_csv(f)
    assert len(df) == 1


def test_solve_single_box():
    df = pd.DataFrame({"item": ["a", "b"], "kg": [0.5, 0.5], "value": [5.0, 5.0]})
    df_result, summary, num_boxes = solve(df, max_kg=2.0, max_value=20.0)
    assert num_boxes == 1
    assert df_result is not None
    assert "box" in df_result.columns


def test_solve_multiple_boxes():
    df = pd.DataFrame({
        "item": ["a", "b"],
        "kg": [1.5, 1.5],
        "value": [5.0, 5.0],
    })
    df_result, summary, num_boxes = solve(df, max_kg=2.0, max_value=20.0)
    assert num_boxes == 2


def test_solve_infeasible():
    # Single item exceeds max_kg — no feasible solution
    df = pd.DataFrame({"item": ["heavy"], "kg": [10.0], "value": [1.0]})
    df_result, summary, num_boxes = solve(df, max_kg=1.0, max_value=100.0)
    assert df_result is None
    assert num_boxes == 0
