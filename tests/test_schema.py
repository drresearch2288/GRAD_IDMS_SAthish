"""Adapters project each domain into the frozen 20-feature schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import (
    COMMON_FEATURES,
    extract_unified_features,
    map_apa_ddos,
    map_bot_iot,
    map_cicids2017,
    map_nsl_kdd,
    map_ton_iot,
    map_unsw_nb15,
)


def _assert_schema(out: pd.DataFrame) -> None:
    assert list(out.columns) == COMMON_FEATURES
    assert len(COMMON_FEATURES) == 20
    assert out.dtypes.eq(np.float32).all()
    assert np.isfinite(out.to_numpy()).all()


def test_map_nsl_kdd_approximates_packets_from_count():
    df = pd.DataFrame(
        {
            "duration": [1.0, 2.0],
            "src_bytes": [10, 20],
            "dst_bytes": [5, 0],
            "count": [3, 8],
            "srv_count": [1, 2],
            "protocol_type": ["tcp", "udp"],
            "service": ["http", "private"],
            "flag": ["SF", "S0"],
            "serror_rate": [0.0, 1.0],
            "rerror_rate": [0.0, 0.0],
        }
    )
    out = map_nsl_kdd(df)
    _assert_schema(out)
    assert out.loc[0, "proto_tcp"] == 1.0
    assert out.loc[1, "proto_udp"] == 1.0
    assert out.loc[1, "error_rate"] > 0.4


def test_map_unsw_nb15_uses_dur_sbytes_spkts():
    df = pd.DataFrame(
        {
            "dur": [0.5],
            "sbytes": [100],
            "dbytes": [40],
            "spkts": [4],
            "dpkts": [2],
            "proto": ["tcp"],
            "service": ["http"],
            "sinpkt": [0.01],
            "dinpkt": [0.02],
            "swin": [1],
            "dwin": [1],
            "state": ["FIN"],
        }
    )
    out = map_unsw_nb15(df)
    _assert_schema(out)
    assert "response_body_len" not in out.columns


def test_map_ton_iot_conn_state_and_ts():
    df = pd.DataFrame(
        {
            "duration": [1.0, 1.0],
            "src_bytes": [10, 10],
            "dst_bytes": [10, 10],
            "src_pkts": [2, 2],
            "dst_pkts": [2, 2],
            "proto": ["tcp", "tcp"],
            "dst_port": [80, 443],
            "conn_state": ["SF", "S0"],
            "ts": [1.0, 1.05],
        }
    )
    out = map_ton_iot(df)
    _assert_schema(out)
    assert out.loc[1, "error_rate"] >= out.loc[0, "error_rate"]


def test_map_bot_iot_excludes_identity_columns():
    df = pd.DataFrame(
        {
            "dur": [0.2],
            "sbytes": [50],
            "dbytes": [10],
            "spkts": [5],
            "dpkts": [1],
            "proto": ["tcp"],
            "dport": [80],
            "flgs": ["e s A P"],
            "state": ["CON"],
            "stime": [100.0],
            "ltime": [100.2],
            "pkSeqID": [99],
            "seq": [7],
        }
    )
    out = map_bot_iot(df)
    _assert_schema(out)
    for banned in ("pkSeqID", "seq", "stime", "ltime"):
        assert banned not in out.columns


def test_map_cicids2017_flag_ratios():
    df = pd.DataFrame(
        {
            "flow_duration": [1e6],
            "total_fwd_packets": [10],
            "total_backward_packets": [10],
            "total_length_of_fwd_packets": [100],
            "total_length_of_bwd_packets": [100],
            "flow_iat_mean": [100.0],
            "flow_iat_min": [1.0],
            "ack_flag_count": [10],
            "psh_flag_count": [2],
            "syn_flag_count": [1],
            "fin_flag_count": [1],
            "rst_flag_count": [0],
            "destination_port": [80],
        }
    )
    out = map_cicids2017(df)
    _assert_schema(out)
    assert 0.4 < float(out.loc[0, "ack_flag_ratio"]) <= 1.0


def test_map_apa_ddos_keeps_tcp_flags():
    df = pd.DataFrame(
        {
            "duration": [1.0],
            "Tx Bytes": [80],
            "Rx Bytes": [20],
            "Tx Packets": [8],
            "Rx Packets": [2],
            "tcp.flags.ack": [8],
            "tcp.flags.push": [4],
            "tcp.flags.syn": [1],
            "tcp.flags.reset": [0],
        }
    )
    out = map_apa_ddos(df)
    _assert_schema(out)
    assert float(out.loc[0, "psh_flag_ratio"]) > 0.0
    assert float(out.loc[0, "ack_flag_ratio"]) > 0.0


def test_extract_unified_features_passthrough_and_dispatch():
    df = pd.DataFrame({n: [0.1, 0.2] for n in COMMON_FEATURES})
    out = extract_unified_features(df, "nsl_kdd")
    _assert_schema(out)
    raw = pd.DataFrame({"duration": [1.0], "src_bytes": [1], "dst_bytes": [1], "count": [1], "srv_count": [1]})
    mapped = extract_unified_features(raw, "nsl_kdd")
    _assert_schema(mapped)
