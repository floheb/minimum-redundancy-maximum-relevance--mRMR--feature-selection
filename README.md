# mRMR Feature Selection

A Python implementation of minimum Redundancy Maximum Relevance (mRMR) feature selection for multiple targets.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

## Usage

The script expects two CSV files with matching ID columns for joining:

- `X_raw.csv`: Features dataset with ID column
- `y_raw.csv`: Targets dataset with ID column

Ids must match

Configure the constants at the top of the script:

```python
X_PATH = "X_raw.csv"
Y_PATH = "y_raw.csv" 
X_ID = "id_x"
Y_ID = "id_y"
N_FEATURES = 3
OUT_PATH = "top3_features.json"

COLUMNS_TO_AVOID_X = [X_ID]
COLUMNS_TO_AVOID_Y = [Y_ID, "IDDR"]
```

Run the selection:

```bash
python main.py
```

## Output

The script generates a JSON file containing the selected features ranked by mRMR score:

```json
{
  "top_features": [
    "feature_name_1",
    "feature_name_2", 
    "feature_name_3"
  ]
}
```

## Features

- Automatic detection of classification vs regression targets
- Missing value imputation (median for numeric, mode for categorical)
- Caching system for redundancy calculations
- Support for multiple target variables
- Colored logging for progress tracking
