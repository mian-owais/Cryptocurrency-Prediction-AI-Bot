from eval_utils import plot_confusion_matrix
import pandas as pd
import sys
from pathlib import Path

# Ensure src is on path
project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))


# Create small mock dataframe
df = pd.DataFrame({
    'actual_label': ['Increase', 'Decrease', 'Constant', 'Increase', 'Decrease'],
    'predicted_label': ['Increase', 'Increase', 'Constant', 'Decrease', 'Decrease']
})

fig = plot_confusion_matrix(df)
print('OK:', type(fig))
