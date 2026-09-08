"""Where a saved model lives on disk, and how it is loaded once.

Four modules here (`persistence_model`, `nbeats_model`, `dayrange_model`,
`momentum_change_model`) wrap a bundle the FinNotebooks projects produce, and
each had grown its own identical copy of the same three concerns: resolve a
ticker to a path under two environment overrides, load it behind a lock, and
cache the result -- including the failure -- so a loop that asks every minute
does not re-hit the filesystem. This is that code, once.

What is NOT here is what each bundle *is*. A `ModelStore` is handed a `build`
callable and never inspects what comes back, because the four differ in every
way that matters: one unpickles a scikit-learn pipeline, one assembles five
networks and a residual matrix, one needs two `.pt` checkpoints beside its
joblib. Validation lives with the module that knows what a valid bundle looks
like -- returning None from `build` is how it refuses.

The two environment overrides
-----------------------------
`<ENV>_<TICKER>` relocates one ticker's file. The bare `<ENV>` names a single
file, so it can only mean the default ticker's: letting it answer for every
symbol would hand a GOOGL run the AAPL model without saying so. Every model
here is fitted per symbol and none of them claims to transfer, so that
distinction is load-bearing rather than pedantic.

The override key is always upper-case (`APPLE_PRICERANGE_MODEL_GOOG`) even for
a family whose *files* are lower-case, because an environment variable is typed
by a person and a filename is written by a notebook. Those are two different
naming authorities, and only the second one gets to be inconsistent.
"""
from __future__ import annotations

import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from .market_hours import MARKET_CLOSE, MARKET_OPEN

# The symbol every model here defaults to, and what a request arriving without
# one means. Mirrors `apple_models.DEFAULT_TICKER`, which is the app-facing
# authority; defined separately because `apple_models` imports the model
# modules and those import this one, so reading it from there would be a cycle.
DEFAULT_TICKER = "AAPL"

# Where the FinNotebooks projects write their bundles: a sibling of this repo,
# so a retrain lands where the app already looks without either side
# configuring the other.
MODEL_DIR = Path(__file__).resolve().parents[2] / "Models"

# The regular-session window the notebooks trained on, as `between_time`
# strings. Derived from the market clock rather than typed, but note the end:
# it is the last minute *bar*, 15:59, not the 16:00 bell. A bar is stamped with
# the minute it opened, so `between_time("09:30", "16:00")` would include a
# 16:00 bar that only exists on a feed that prints the closing auction -- and
# `mshift.data` / `momlib.data` both cut at 15:59, so a live frame built the
# other way would carry a bar no training frame ever had.
RTH_START = MARKET_OPEN.strftime("%H:%M")
RTH_END = (
    datetime.combine(date.min, MARKET_CLOSE) - timedelta(minutes=1)
).strftime("%H:%M")


class ModelStore:
    """One model family's files, resolved per ticker and loaded at most once.

    `filename` is a template taking `{ticker}`, e.g.
    ``"timetochange2_persistence_{ticker}.joblib"``. It deliberately mirrors
    the name the notebook saves under -- the two stores agreeing is what makes
    a retrain visible to the app with no further step. `lowercase_file` is part
    of that mirroring rather than a style preference: PriceRange2 builds its
    filename from `config.MODEL_NAME` and so writes `pricerange2_aapl.joblib`.
    A store that upper-cased the symbol would look in the wrong place, and
    would go on looking in the wrong place after every future retrain.
    """

    def __init__(
        self,
        env_key: str,
        filename: str,
        build: "Callable[[Path], dict | None]",
        *,
        model_dir: "Path | None" = None,
        default_ticker: str = DEFAULT_TICKER,
        lowercase_file: bool = False,
    ) -> None:
        self.env_key = env_key
        self.filename = filename
        self.default_ticker = default_ticker
        self.lowercase_file = lowercase_file
        self.model_dir = model_dir or MODEL_DIR
        self._build = build
        self._lock = threading.Lock()
        # Keyed by path rather than one slot, so a process that runs GOOGL
        # after AAPL does not evict and re-load a bundle each time it switches.
        # Failures cache under the same key, as None.
        self._cache: "dict[Path, dict | None]" = {}

    @property
    def per_ticker(self) -> bool:
        """Whether this family has one file per symbol.

        Read off the filename template rather than configured separately, so
        the two cannot disagree. A single-file model (one fitted on pooled
        data, or on a feature every symbol shares) takes the bare env override
        only -- offering it a `<ENV>_<TICKER>` that silently does nothing would
        be worse than not offering one.
        """
        return "{ticker}" in self.filename

    def path(self, ticker: "str | None" = None) -> Path:
        """Where one ticker's file is expected to live (see module docstring)."""
        if not self.per_ticker:
            return Path(os.environ.get(self.env_key) or self.model_dir / self.filename)
        symbol = (ticker or self.default_ticker).upper()
        override = os.environ.get(f"{self.env_key}_{symbol}")
        if not override and symbol == self.default_ticker:
            override = os.environ.get(self.env_key)
        # The env key keeps the upper-case symbol; only the filename follows
        # whatever case the notebook that wrote it chose.
        stem = symbol.lower() if self.lowercase_file else symbol
        return Path(override or self.model_dir / self.filename.format(ticker=stem))

    def metadata_path(self, path: "Path | None" = None) -> Path:
        """The readable JSON sidecar the save step writes beside the model."""
        return Path(path or self.path()).with_suffix(".json")

    def sidecar_path(self, suffix: str, path: "Path | None" = None) -> Path:
        """A companion file sharing the model's stem, e.g. `_residuals.npz`."""
        base = Path(path or self.path())
        return base.with_name(f"{base.stem}{suffix}")

    def load(self, ticker: "str | None" = None) -> "dict | None":
        """The built bundle for one ticker, or None when it cannot be had.

        Cached per path after the first attempt, the failure included: a model
        that is not installed is not installed, and re-checking once a minute
        for the life of the process buys nothing.
        """
        target = self.path(ticker)
        with self._lock:
            if target in self._cache:
                return self._cache[target]
            bundle = self._build(target)
            self._cache[target] = bundle
            return bundle

    def reset(self) -> None:
        """Forget every loaded bundle -- for tests that move the model path."""
        with self._lock:
            self._cache.clear()
