import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from rulekit import RuleKit
from rulekit.classification import RuleClassifier, Measures

class RulekitWrapper:
    def __init__(
        self,
        minsupp_new: float = 0.05,
        induction_measure=Measures.C2,
        pruning_measure=Measures.C2,
        voting_measure=Measures.C2,
        max_growing: int = 0,
        max_rule_count: int = 0,
        enable_pruning: bool = False,
        ignore_missing: bool = False,
        approximate_induction: bool = False,
        approximate_bins_count: int = 7
    ):
        RuleKit.init()
        self.clf = RuleClassifier(
            minsupp_new=minsupp_new,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_rule_count=max_rule_count,
            approximate_induction=approximate_induction,
            approximate_bins_count=approximate_bins_count
        )
        self.is_fitted = False

    # ---------------------------
    # Utility Functions
    # ---------------------------

    def _parse_rule_text(self, rule_text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Parse a rule into conditions + predicted label."""
        rule_text = rule_text.strip()
        m = re.search(r'\bIF\b(.*)\bTHEN\b(.*)', rule_text, flags=re.IGNORECASE)
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
        else:
            parts = rule_text.split('THEN')
            left = parts[0] if parts else rule_text
            right = parts[1] if len(parts) > 1 else None

        predicted = None
        if right:
            mm = re.search(r'\{([^}]*)\}', right)
            if mm:
                predicted = mm.group(1).strip()

        cond_texts = re.split(r'\s+\bAND\b\s+', left, flags=re.IGNORECASE)
        conds = []
        for ctext in cond_texts:
            c = ctext.strip()
            if not c:
                continue
            parts = re.split(r'\s*=\s*', c, maxsplit=1)
            if len(parts) != 2:
                conds.append({'attr': None, 'low': None, 'high': None,
                              'inclusive_low': False, 'inclusive_high': False,
                              'raw': c})
                continue
            attr = parts[0].strip()
            interval = parts[1].strip()

            interval_inner = interval[1:].strip() if interval.startswith(('(', '[')) else interval
            if interval_inner.endswith((')', ']')):
                interval_inner = interval_inner[:-1].strip()

            if ',' in interval_inner:
                left_tok, right_tok = [t.strip() for t in interval_inner.split(',', 1)]
            else:
                left_tok, right_tok = interval_inner, None

            def _parse_bound(tok: Optional[str]) -> Tuple[Optional[float], bool]:
                if tok is None:
                    return None, False
                inclusive = ('[' in tok) or ('<=' in tok) or ('≤' in tok)
                t = re.sub(r'[^\d\.\-+eE]', '', tok)
                if t == '' or t.lower().startswith('inf'):
                    return None, inclusive
                try:
                    return float(t), inclusive
                except Exception:
                    return None, inclusive

            low_val, incl_low = _parse_bound(left_tok)
            high_val, incl_high = _parse_bound(right_tok)

            conds.append({
                'attr': attr,
                'low': low_val,
                'high': high_val,
                'inclusive_low': incl_low,
                'inclusive_high': incl_high,
                'raw': c
            })
        return conds, predicted

    def _condition_mask(self, cond: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
        attr = cond['attr']
        if attr not in df.columns:
            return pd.Series(False, index=df.index)
        series = df[attr].astype(float)
        low = cond['low']
        high = cond['high']
        il = cond.get('inclusive_low', False)
        ih = cond.get('inclusive_high', False)

        mask = pd.Series(True, index=df.index)
        if low is not None:
            mask &= (series >= low) if il else (series > low)
        if high is not None:
            mask &= (series <= high) if ih else (series < high)
        mask &= series.notna()
        return mask

    def _parse_rules(self, rules_list: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        parsed_rules = []
        for r in rules_list:
            t, w = r if isinstance(r, tuple) else (r, 1.0)
            conds, pred = self._parse_rule_text(t)
            parsed_rules.append({'text': t, 'conds': conds, 'predicted': pred, 'weight': w})
        return parsed_rules

    @staticmethod
    def extract_rule_label(rule_text: str):
        """Extract predicted label from rule text."""
        match = re.search(r"\{\s*([0-9]+(?:\.[0-9]+)?)\s*\}", rule_text)
        if not match:
            raise ValueError(f"Could not extract label from rule: {rule_text}")
        label_str = match.group(1)
        try:
            label = int(float(label_str))
        except ValueError:
            label = float(label_str)
        return label

    # ---------------------------
    # Core API
    # ---------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.clf.fit(X, y)
        self.is_fitted = True

    def get_rules(self) -> List[Tuple[str, float]]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet!")
        return [(str(rule), rule.weight) for rule in self.clf.model.rules]

    def predict(self, X: pd.DataFrame, rules_list: List[Tuple[str, float]] = None) -> np.ndarray:
        """Predict labels using weighted majority voting."""
        if rules_list is None:
            if not self.is_fitted:
                raise ValueError("Model is not fitted and no rules provided!")
            rules_list = self.get_rules()

        parsed = self._parse_rules(rules_list)
        rule_labels = [self.extract_rule_label(rule["text"]) for rule in parsed]
        unique_labels = sorted(set(rule_labels))

        y_pred = []
        for idx in X.index:
            label_votes = {}
            for rule, label in zip(parsed, rule_labels):
                mask = pd.Series(True, index=[idx])
                for cond in rule['conds']:
                    mask &= self._condition_mask(cond, X.loc[[idx]])
                if mask.iloc[0]:
                    label_votes[label] = label_votes.get(label, 0) + rule['weight']

            if not label_votes:
                # No rule applies → default to majority label
                pred_label = max(unique_labels, key=lambda l: rule_labels.count(l))
            else:
                # Pick label with highest total weight
                pred_label = max(label_votes, key=label_votes.get)

            y_pred.append(pred_label)

        return np.array(y_pred)

    def local_explainability(self, x: pd.Series, rules_list: Optional[List[Tuple[str, float]]] = None) -> dict:
        if rules_list is None:
            if not self.is_fitted:
                raise RuntimeError("Model not fitted and no rules provided!")
            rules_list = self.get_rules()

        parsed = self._parse_rules(rules_list)
        rule_labels = [self.extract_rule_label(rule["text"]) for rule in parsed]

        matched_rules = []
        for i, rule in enumerate(parsed):
            mask = True
            for cond in rule['conds']:
                mask &= self._condition_mask(cond, x.to_frame().T).iloc[0]
            if mask:
                matched_rules.append({
                    "rule_index": i,
                    "rule_text": rule["text"],
                    "rule_quality": rule["weight"],
                    "predicted_label": rule_labels[i]
                })

        if len(matched_rules) == 0:
            unique_labels = list(set(rule_labels))
            return {"predicted_label": random.choice(unique_labels), "matched_rules": []}

        votes, quality_sums = {}, {}
        for r in matched_rules:
            lbl = r["predicted_label"]
            q = r["rule_quality"]
            votes[lbl] = votes.get(lbl, 0) + 1
            quality_sums[lbl] = quality_sums.get(lbl, 0) + q

        max_votes = max(votes.values())
        top_labels = [lbl for lbl, v in votes.items() if v == max_votes]
        predicted_label = max(top_labels, key=lambda l: quality_sums[l]) if len(top_labels) > 1 else top_labels[0]

        return {"predicted_label": predicted_label, "matched_rules": matched_rules}

    def rules_statistics(self, X: pd.DataFrame, y: pd.Series, rules_list: Optional[List[Tuple[str, float]]] = None) -> pd.DataFrame:
        if rules_list is None:
            if not self.is_fitted:
                raise RuntimeError("Model not fitted and no rules provided!")
            rules_list = self.get_rules()

        parsed = self._parse_rules(rules_list)
        rows = []
        for i, rule in enumerate(parsed):
            mask = pd.Series(True, index=X.index)
            for cond in rule['conds']:
                mask &= self._condition_mask(cond, X)

            predicted_label = self.extract_rule_label(rule["text"])
            P = int((y == predicted_label).sum())
            N = int((y != predicted_label).sum())
            covered_idx = mask[mask].index
            p = int((y.loc[covered_idx] == predicted_label).sum())
            n = int((y.loc[covered_idx] != predicted_label).sum())
            rule_quality = (p / P if P > 0 else 0) * (1 - (n / N if N > 0 else 0))

            rows.append({
                "rule_index": i,
                "rule_text": rule['text'],
                "predicted_label": predicted_label,
                "weight": rule['weight'],
                "rule_quality": round(rule_quality, 4),
                "p": p, "n": n, "P": P, "N": N,
                "covered_examples": int(mask.sum()),
                "covered_fraction": round((p+n)/(P+N), 4)
            })

        return pd.DataFrame(rows)
