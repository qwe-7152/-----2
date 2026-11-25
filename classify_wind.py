import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "wind_dataset.csv"

PREDICTION_TIME_COLUMN = "prediction_time"
TARGET_TIME_COLUMN = "target_time"
NEXT_WIND_COLUMN = "target_wind_speed_ms"
LABEL_COLUMN = "label"
NEXT_WIND_ALIAS_COLUMN = "wind_next1h"
FEATURE_COLUMNS = [
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_1h_ago_ms",
    "wind_speed_2h_ago_ms",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    required = {
        PREDICTION_TIME_COLUMN,
        TARGET_TIME_COLUMN,
        NEXT_WIND_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(sorted(missing))}")

    df[PREDICTION_TIME_COLUMN] = pd.to_datetime(
        df[PREDICTION_TIME_COLUMN], errors="raise"
    )
    df[TARGET_TIME_COLUMN] = pd.to_datetime(
        df[TARGET_TIME_COLUMN], errors="raise"
    )
    df[NEXT_WIND_COLUMN] = pd.to_numeric(
        df[NEXT_WIND_COLUMN], errors="raise"
    )
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df = df.sort_values(
        [TARGET_TIME_COLUMN, PREDICTION_TIME_COLUMN],
        kind="stable",
    ).reset_index(drop=True)

    differences = df[TARGET_TIME_COLUMN] - df[PREDICTION_TIME_COLUMN]
    invalid = df.index[differences != pd.Timedelta(hours=1)].tolist()
    if invalid:
        rows = invalid[:10]
        raise ValueError(
            "예측 시각과 정답 시각의 차이가 정확히 1시간이 아닌 행이 있습니다 "
            f"(처음 10개 행 index: {rows})"
        )

    df[NEXT_WIND_ALIAS_COLUMN] = df[NEXT_WIND_COLUMN]
    df[LABEL_COLUMN] = (df[NEXT_WIND_ALIAS_COLUMN] >= 4.0).astype(int)
    return df


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_dataset()

    train_mask = (
        (df[TARGET_TIME_COLUMN] >= pd.Timestamp("2023-01-01 00:00:00"))
        & (df[TARGET_TIME_COLUMN] < pd.Timestamp("2024-12-01 00:00:00"))
    )
    eval_mask = (
        (df[TARGET_TIME_COLUMN] >= pd.Timestamp("2024-12-01 00:00:00"))
        & (df[TARGET_TIME_COLUMN] < pd.Timestamp("2025-01-01 00:00:00"))
    )

    train_df = df.loc[train_mask].copy()
    eval_df = df.loc[eval_mask].copy()

    outside_mask = ~(train_mask | eval_mask)
    if outside_mask.any():
        first = df.loc[outside_mask, TARGET_TIME_COLUMN].min()
        last = df.loc[outside_mask, TARGET_TIME_COLUMN].max()
        raise ValueError(
            "학습 또는 평가 기간에 속하지 않는 정답 시각이 있습니다 "
            f"(범위: {first} ~ {last}, 행 수: {outside_mask.sum()})"
        )

    train_features = train_df[FEATURE_COLUMNS]
    eval_features = eval_df[FEATURE_COLUMNS]
    if train_features.isna().any().any() or eval_features.isna().any().any():
        raise ValueError("모델 입력 특성에 결측치가 있습니다.")

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    eval_features_scaled = scaler.transform(eval_features)

    model = LogisticRegression(
        class_weight=None,
        random_state=42,
        max_iter=1000,
    )
    model.fit(train_features_scaled, train_df[LABEL_COLUMN])

    strong_wind_probability = model.predict_proba(eval_features_scaled)[:, 1]
    logistic_prediction = (strong_wind_probability >= 0.50).astype(int)
    logistic_prediction_020 = (strong_wind_probability >= 0.20).astype(int)

    predictions_df = pd.DataFrame(
        {
            PREDICTION_TIME_COLUMN: eval_df[PREDICTION_TIME_COLUMN].to_numpy(),
            TARGET_TIME_COLUMN: eval_df[TARGET_TIME_COLUMN].to_numpy(),
            "actual_wind_speed_ms": eval_df[NEXT_WIND_COLUMN].to_numpy(),
            "label": eval_df[LABEL_COLUMN].to_numpy(),
            "current_wind_speed_ms": eval_df["wind_speed_ms"].to_numpy(),
            "always_zero_prediction": 0,
            "current_wind_threshold_prediction": (
                eval_df["wind_speed_ms"].to_numpy() >= 4.0
            ).astype(int),
            "strong_wind_probability": strong_wind_probability,
            "logistic_prediction": logistic_prediction,
            "logistic_prediction_0.20": logistic_prediction_020,
        }
    )
    predictions_df.to_csv(
        BASE_DIR / "classification_predictions.csv",
        index=False,
    )

    same_eval_rows = (
        len(predictions_df) == len(eval_df)
        and (
            predictions_df[PREDICTION_TIME_COLUMN].to_numpy()
            == eval_df[PREDICTION_TIME_COLUMN].to_numpy()
        ).all()
        and (
            predictions_df[TARGET_TIME_COLUMN].to_numpy()
            == eval_df[TARGET_TIME_COLUMN].to_numpy()
        ).all()
        and (
            predictions_df["actual_wind_speed_ms"].to_numpy()
            == eval_df[NEXT_WIND_COLUMN].to_numpy()
        ).all()
        and (
            predictions_df["label"].to_numpy()
            == eval_df[LABEL_COLUMN].to_numpy()
        ).all()
    )

    actual = predictions_df["label"].to_numpy()
    method_columns = [
        ("always_zero", "항상0"),
        ("current_wind_threshold", "현재풍속4.0 m/s 이상"),
        ("logistic_050", "로지스틱회귀(0.50)"),
        ("logistic_020", "로지스틱회귀(0.20)"),
    ]
    prediction_columns = {
        "always_zero": "always_zero_prediction",
        "current_wind_threshold": "current_wind_threshold_prediction",
        "logistic_050": "logistic_prediction",
        "logistic_020": "logistic_prediction_0.20",
    }

    score_rows = []
    for method_key, method_name in method_columns:
        predicted = predictions_df[prediction_columns[method_key]].to_numpy()
        hit = int(((predicted == 1) & (actual == 1)).sum())
        miss = int(((predicted == 0) & (actual == 1)).sum())
        false_alarm = int(((predicted == 1) & (actual == 0)).sum())
        correct_normal = int(((predicted == 0) & (actual == 0)).sum())
        predicted_strong = hit + false_alarm
        actual_strong = hit + miss

        accuracy = (hit + correct_normal) / len(actual)
        precision = hit / predicted_strong if predicted_strong > 0 else None
        recall = hit / actual_strong if actual_strong > 0 else None

        score_rows.append(
            {
                "method": method_name,
                "evaluation_rows": len(actual),
                "actual_strong": actual_strong,
                "hits": hit,
                "misses": miss,
                "false_alarms": false_alarm,
                "correct_normals": correct_normal,
                "accuracy": accuracy,
                "precision": (
                    precision
                    if precision is not None
                    else "해당없음"
                ),
                "recall": recall,
            }
        )

    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(
        BASE_DIR / "classification_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    week_start = pd.Timestamp("2024-12-01 00:00:00")
    week_end = pd.Timestamp("2024-12-08 00:00:00")
    december_week_df = predictions_df[
        (predictions_df[TARGET_TIME_COLUMN] >= week_start)
        & (predictions_df[TARGET_TIME_COLUMN] < week_end)
    ].copy()

    dec2_start = pd.Timestamp("2024-12-02 00:00:00")
    dec2_end = pd.Timestamp("2024-12-03 00:00:00")
    dec2_wrong_df = december_week_df[
        (december_week_df[TARGET_TIME_COLUMN] >= dec2_start)
        & (december_week_df[TARGET_TIME_COLUMN] < dec2_end)
        & (
            (december_week_df["logistic_prediction"]
             != december_week_df["label"])
            | (december_week_df["logistic_prediction_0.20"]
               != december_week_df["label"])
        )
    ]

    fig, (wind_ax, prediction_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1]},
    )

    wind_ax.plot(
        december_week_df[TARGET_TIME_COLUMN],
        december_week_df["actual_wind_speed_ms"],
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="실제 풍속",
    )
    wind_ax.axhline(
        4.0,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="4.0 m/s 기준선",
    )
    wind_ax.set_ylabel("풍속 (m/s)")
    wind_ax.set_title("12월 1~7일 실제 풍속과 강풍 판정")
    wind_ax.legend(loc="upper right")
    wind_ax.grid(True, alpha=0.3)

    actual_strong = december_week_df[
        december_week_df["label"] == 1
    ]
    logistic_050_strong = december_week_df[
        december_week_df["logistic_prediction"] == 1
    ]
    logistic_020_strong = december_week_df[
        december_week_df["logistic_prediction_0.20"] == 1
    ]

    prediction_ax.scatter(
        actual_strong[TARGET_TIME_COLUMN],
        [3] * len(actual_strong),
        color="#9467bd",
        marker="o",
        s=32,
        label="실제 강풍",
    )
    prediction_ax.scatter(
        logistic_050_strong[TARGET_TIME_COLUMN],
        [2] * len(logistic_050_strong),
        color="#ff7f0e",
        marker="s",
        s=32,
        label="로지스틱 0.50 예측",
    )
    prediction_ax.scatter(
        logistic_020_strong[TARGET_TIME_COLUMN],
        [1] * len(logistic_020_strong),
        color="#2ca02c",
        marker="^",
        s=32,
        label="로지스틱 0.20 예측",
    )
    prediction_ax.set_yticks([1, 2, 3])
    prediction_ax.set_yticklabels(
        ["0.20 예측", "0.50 예측", "실제 강풍"]
    )
    prediction_ax.set_xlabel("정답 시각 (KST)")
    prediction_ax.set_ylabel("강풍 판정")
    prediction_ax.legend(loc="upper right", ncol=3)
    prediction_ax.grid(True, axis="x", alpha=0.3)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(
        BASE_DIR / "strong_wind_december.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    train_df.to_csv(BASE_DIR / "wind_train.csv", index=False)
    eval_df.to_csv(BASE_DIR / "wind_eval.csv", index=False)
    df.to_csv(BASE_DIR / "wind_dataset_labeled.csv", index=False)

    print("시각 기준: Asia/Seoul (KST)")
    print("예측 시각과 정답 시각 차이 검증: 모든 행에서 정확히 1시간")
    print(f"학습 전체 행 수: {len(train_df)}")
    print(f"학습 0 개수: {(train_df[LABEL_COLUMN] == 0).sum()}")
    print(f"학습 1 개수: {(train_df[LABEL_COLUMN] == 1).sum()}")
    print(f"평가 전체 행 수: {len(eval_df)}")
    print(f"평가 0 개수: {(eval_df[LABEL_COLUMN] == 0).sum()}")
    print(f"평가 1 개수: {(eval_df[LABEL_COLUMN] == 1).sum()}")
    print(
        "평가 행 일치 확인: "
        + ("예, 예측 CSV와 동일한 744개 평가 행을 사용" if same_eval_rows else "아니오")
    )
    print(f"실제 강풍 수: {(eval_df[LABEL_COLUMN] == 1).sum()}")
    print("방법별 결과:")
    display_scores = scores_df.rename(
        columns={
            "method": "방법",
            "evaluation_rows": "평가행수",
            "actual_strong": "실제강풍수",
            "hits": "적중",
            "misses": "놓침",
            "false_alarms": "오경보",
            "correct_normals": "정상판정",
            "accuracy": "정확도",
            "precision": "강풍정밀도",
            "recall": "강풍재현율",
        }
    )
    print(
        display_scores.to_string(
            index=False,
            formatters={
                "정확도": lambda value: f"{value:.6f}",
                "강풍정밀도": lambda value: (
                    value if isinstance(value, str) else f"{value:.6f}"
                ),
                "강풍재현율": lambda value: f"{value:.6f}",
            },
        )
    )
    threshold_comparison = display_scores[
        display_scores["방법"].isin(
            ["로지스틱회귀(0.50)", "로지스틱회귀(0.20)"]
        )
    ][["방법", "적중", "놓침", "오경보", "강풍정밀도", "강풍재현율"]]
    print("0.50과 0.20 판정 비교:")
    print(threshold_comparison.to_string(index=False))
    print(
        f"12월 1~7일 그래프 행 수: {len(december_week_df)}"
    )
    print("12월 2일 틀린 사례:")
    if dec2_wrong_df.empty:
        print("없음")
    else:
        first_wrong = dec2_wrong_df.iloc[0]
        print(
            first_wrong[
                [
                    PREDICTION_TIME_COLUMN,
                    TARGET_TIME_COLUMN,
                    "actual_wind_speed_ms",
                    "label",
                    "current_wind_speed_ms",
                    "strong_wind_probability",
                    "logistic_prediction",
                    "logistic_prediction_0.20",
                ]
            ].to_string()
        )

    if eval_df.empty:
        print("첫 평가 행: 없음")
    else:
        print("첫 평가 행:")
        print(eval_df.iloc[0].to_string())
        print("예측 CSV 첫 5행:")
        print(predictions_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
