import os
from pathlib import Path

pathbase = str(Path(os.getcwd()).parents[0])
print("path base ", pathbase)