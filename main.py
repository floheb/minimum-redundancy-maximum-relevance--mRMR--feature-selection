"""
mRMR Feature Selection Module
============================

Implements minimum redundancy maximum relevance (mRMR) feature selection for multiple targets with 
on-demand redundancy computation and validity masking for missing values.

"""

import json
import sys
from pathlib import Path
from logger import ColoredLogger  
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from tqdm import tqdm

# ─────────────────────────────── CONSTANTS ──────────────────────────────────

X_PATH: str = "X_raw.csv"
Y_PATH: str = "y_raw.csv" 
X_ID: str = "id_x"
Y_ID: str = "id_y"
N_FEATURES: int = 3
OUT_PATH: str = "top3_features.json"

COLUMNS_TO_AVOID_X = [X_ID]
COLUMNS_TO_AVOID_Y = [Y_ID, "IDDR"]

# Global logger instance
logger = ColoredLogger()

# ─────────────────────────────── GLOBAL CACHE ───────────────────────────────

_pair_mi_cache: dict[tuple[str, str], float] = {}

# ─────────────────────────────── UTILITY FUNCTIONS ──────────────────────────

def _impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using median for numeric and mode for categorical columns.
    Returns the imputed DataFrame to prevent feature-feature MI computation failures.
    """
    logger.info(f"Imputing missing values in features (shape {df.shape})")
    df_imputed = df.copy()
    
    for col in df_imputed.columns:
        if pd.api.types.is_numeric_dtype(df_imputed[col]):
            df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
        else:
            mode_values = df_imputed[col].mode(dropna=True)
            if len(mode_values) > 0:
                df_imputed[col].fillna(mode_values[0], inplace=True)
    
    return df_imputed

def _compute_feature_target_mi(feature: pd.Series, targets: pd.DataFrame) -> float:
    """
    Compute average mutual information between a feature and all target columns.
    Uses classification MI for binary integer/boolean targets, regression MI otherwise.
    """
    mi_scores: list[float] = []
    
    for target_col in targets.columns:
        target = targets[target_col]
        valid_mask = (~feature.isna()) & (~target.isna())
        
        if valid_mask.sum() == 0:
            mi_scores.append(0.0)
            logger.error(f"No valid overlap: feature={feature.name}, target={target_col}")
            continue
        
        feature_valid = np.asarray(feature[valid_mask].values).reshape(-1, 1)
        target_valid = np.asarray(target[valid_mask].values)
        
        # Determine MI type based on target characteristics
        unique_values = np.unique(target_valid)
        is_binary_discrete = (
            len(unique_values) == 2 and 
            (pd.api.types.is_integer_dtype(target.dtype) or 
             pd.api.types.is_bool_dtype(target.dtype))
        )
        
        if is_binary_discrete:
            mi_score = mutual_info_classif(
                feature_valid, target_valid, discrete_features=False
            )[0]
        else:
            mi_score = mutual_info_regression(
                feature_valid, target_valid, discrete_features=False
            )[0]
            logger.info(f"Using regression MI for '{feature.name}' vs '{target_col}'")
        
        mi_scores.append(mi_score)
    
    return float(np.mean(mi_scores))

def _compute_feature_pair_mi(
    feature_i: pd.Series, 
    feature_j: pd.Series,
    name_i: str,
    name_j: str
) -> float:
    """
    Compute mutual information between two features with caching.
    Cache key uses alphabetically sorted feature names for consistency.
    """
    # Ensure cache key is exactly a tuple of two strings
    name_tuple = sorted((name_i, name_j))
    cache_key = (name_tuple[0], name_tuple[1])
    
    if cache_key in _pair_mi_cache:
        return _pair_mi_cache[cache_key]
    
    valid_mask = (~feature_i.isna()) & (~feature_j.isna())
    
    if valid_mask.sum() == 0:
        mi_value = 0.0
    else:
        feature_i_valid = np.asarray(feature_i[valid_mask].values).reshape(-1, 1)
        feature_j_valid = np.asarray(feature_j[valid_mask].values)
        mi_value = mutual_info_regression(
            feature_i_valid, feature_j_valid, discrete_features=False
        )[0]
    
    _pair_mi_cache[cache_key] = mi_value
    
    if len(_pair_mi_cache) % 1000 == 0:
        logger.info(f"Cached {len(_pair_mi_cache)} feature pair MI computations")
    
    return mi_value

# ─────────────────────────────── MRMR IMPLEMENTATION ────────────────────────

def select_mrmr_features(
    features: pd.DataFrame,
    targets: pd.DataFrame, 
    n_select: int = N_FEATURES
) -> list[str]:
    """
    Select top n features using mRMR with on-demand redundancy computation.
    Returns list of selected feature names ordered by selection priority.
    """
    feature_names = features.columns.tolist()
    
    # Compute relevance scores for all features
    logger.info(f"Computing relevance scores for {len(feature_names)} features")
    relevance_scores: dict[str, float] = {
        name: _compute_feature_target_mi(features[name], targets)
        for name in tqdm(feature_names, desc="Computing relevance")
    }
    
    logger.info("Relevance scores computed:")
    for name, score in relevance_scores.items():
        logger.info(f"  {name}: {score:.4f}")
    
    selected_features: list[str] = []
    candidate_features = set(feature_names)
    
    # Iterative feature selection
    for selection_step in range(n_select):
        logger.info(f"Selection step {selection_step + 1}/{n_select}")
        
        best_feature: str | None = None
        best_score = -np.inf
        
        for candidate in candidate_features:
            relevance = relevance_scores[candidate]
            
            # Compute redundancy with already selected features
            if selected_features:
                redundancy_scores = [
                    _compute_feature_pair_mi(
                        features[candidate], features[selected], candidate, selected
                    )
                    for selected in selected_features
                ]
                redundancy = float(np.mean(redundancy_scores))
            else:
                redundancy = 0.0
            
            # mRMR score: relevance minus normalized redundancy
            mrmr_score = relevance - redundancy / max(len(selected_features), 1)
            
            logger.info(
                f"  {candidate:25s} | relevance={relevance:.4f} | "
                f"redundancy={redundancy:.4f} | score={mrmr_score:.4f}"
            )
            
            if mrmr_score > best_score:
                best_feature = candidate
                best_score = mrmr_score
        
        if best_feature is None:
            logger.warning("No improvement found, stopping selection early")
            break
        
        logger.info(f"Selected: {best_feature} (score={best_score:.4f})")
        selected_features.append(best_feature)
        candidate_features.remove(best_feature)
    
    return selected_features

# ─────────────────────────────── MAIN EXECUTION ─────────────────────────────

def main() -> None:
    """Main execution function for mRMR feature selection pipeline."""
    try:
        # Load input data
        logger.info(f"Loading features from {X_PATH}")
        features_df = pd.read_csv(X_PATH)
        
        logger.info(f"Loading targets from {Y_PATH}")
        targets_df = pd.read_csv(Y_PATH)
        
        # Impute missing values in features
        features_df = _impute_features(features_df)
        
        # Prepare target columns (exclude ID columns)
        target_columns = [col for col in targets_df.columns if col not in COLUMNS_TO_AVOID_Y]
        
        # Join datasets on ID columns
        logger.info(f"Joining datasets on {X_ID}/{Y_ID}")
        merged_data = features_df.merge(
            targets_df,
            left_on=X_ID,
            right_on=Y_ID,
            how="inner",
            suffixes=("", "_target")
        )
        
        if merged_data.empty:
            logger.error("No overlapping IDs found after join")
            sys.exit(1)
        
        logger.info(f"Successfully joined {len(merged_data)} records")
        
        # Extract features and targets from merged data
        feature_cols = [col for col in features_df.columns if col not in COLUMNS_TO_AVOID_X]


        X_joined = merged_data[feature_cols]
        y_joined = merged_data[target_columns]
        
        logger.info(f"Final data shapes: X={X_joined.shape}, y={y_joined.shape}")
        
        # Run mRMR feature selection
        selected_features = select_mrmr_features(X_joined, y_joined, N_FEATURES)
        
        # Save results
        output_path = Path(OUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {"top_features": selected_features}
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2)
        
        logger.info(f"Top {N_FEATURES} features saved to {OUT_PATH}")
        logger.info(f"Selected features: {selected_features}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()