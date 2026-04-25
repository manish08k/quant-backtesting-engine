"""models/evaluation.py"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from utils.logger import get_logger

log = get_logger(__name__)


class Evaluator:
    def summarise_folds(self, folds: list[dict]):
        if not folds:
            log.warning("No folds to summarise")
            return
        df = pd.DataFrame(folds)
        log.info(
            f"\n{'='*55}\nWALK-FORWARD SUMMARY\n{'='*55}\n"
            f"{df[['fold','auc','f1','win_rate','threshold']].to_string(index=False)}\n"
            f"Mean AUC={df['auc'].mean():.3f}  F1={df['f1'].mean():.3f}  "
            f"WR={df['win_rate'].mean():.3f}"
        )
