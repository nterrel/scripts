# From ignacio: add this to use different folds for different models in ensemble training

import sys
if sys.argc > 1:
    fold_num = "0"
else:
    fold_num = sys.argv[1]
function(split=f"training{fold}")
