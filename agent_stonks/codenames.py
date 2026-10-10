"""What the screen calls each trained model.

The working names -- TimeToChange3, HighLow, LevelsML and the rest -- say what
each model forecasts, which is more than the app should tell someone watching
it. Everything a user reads names a model through the codenames below instead.
The code, the saved files in `Code/Models`, the `data/` caches, stored SimLab
records and the logs keep the working names, so changing a codename here moves
every label, picker, overlay and help text with it and touches nothing on disk.

Working name                         Key                  Codename
TimeToChange3 (day-range forecast)   `dayrange`           Polaris
IntradayVolatility                   `intraday_vol`       Lyra
Day Range × Intraday Volatility      `dayrange_intraday`  Polaris × Lyra
HighLow_5m                           `highlow`            Orion
HighLow2_5m                          `highlow2`           Orion II
HighLow_3m                           `highlow3m`          Altair
LevelsML (open price profile)        `open_profile`       Helix
NewsImpact                           --                   Sirius
"""

from __future__ import annotations

DAYRANGE = "Polaris"
INTRADAY_VOL = "Lyra"
DAYRANGE_INTRADAY = f"{DAYRANGE} × {INTRADAY_VOL}"
HIGHLOW = "Orion"
HIGHLOW2 = "Orion II"
HIGHLOW3M = "Altair"
OPEN_PROFILE = "Helix"
NEWS_IMPACT = "Sirius"
