"""The tool schemas each personality is given, as the API receives them.

Split out of `agent.py` for the same reason `agent_prompts` was: this is
declarative data describing what the model may call, and it had grown to the
same size as the code that answers those calls. The handlers stay in `agent.py`
next to the state they read; `agent.py`'s `_DISPATCH` is what binds a name here
to a handler there, and a schema without a dispatch entry is a tool the model
can ask for and never get an answer to.

The `*_TOOLS` lists at the bottom are the point of the split: a personality's
capability is one readable list rather than something to be reconstructed by
scrolling nine hundred lines of JSON.
"""
from __future__ import annotations

import copy

from .state import ALERTABLE_FIELDS
from .tactics import TACTIC_CONDITION_FIELDS


_TOOL_GET_QUOTE = {
    "type": "function",
    "function": {
        "name": "get_quote",
        "description": (
            "Get the latest streamed quote and trade price for the ticker, including "
            "spread, spread_pct and quote age. If a `warning` field is present the "
            "bid/ask are unreliable (placeholder-wide or stale off-hours quote from the "
            "thin IEX book) -- trust last_price over bid/ask in that case."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_INTRADAY_MOMENTUM = {
    "type": "function",
    "function": {
        "name": "analyze_intraday_momentum",
        "description": (
            "Analyze recent intraday price action for the ticker: momentum pattern "
            "(higher highs/lows vs lower highs/lows), position relative to session VWAP, "
            "and ATR-based volatility. Also reports yesterday's momentum, today's total "
            "momentum over the full session, how long the current momentum leg has lasted "
            "(via piecewise-linear regime detection), and market-neutral momentum -- the "
            "ticker's move with the beta-scaled broad-market (SPY) move removed. Returns "
            "labeled values plus a one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent bars to analyze (default 50, max 300).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_DAILY_TREND = {
    "type": "function",
    "function": {
        "name": "analyze_daily_trend",
        "description": (
            "Analyze daily bars (up to ~1 year) to establish the medium-term trading regime: "
            "bullish/bearish/neutral with strength, moving-average alignment, RSI, and recent "
            "support/resistance. Returns labeled values plus a one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent daily bars (default 60).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_MARKET = {
    "type": "function",
    "function": {
        "name": "analyze_market",
        "description": (
            "Analyze broad-market conditions (independent of the ticker) using the best-known "
            "regime gauges: the VIX fear level and its trend, the VIX term structure "
            "(near-term vs 3-month implied vol), and the S&P 500's primary trend, drawdown, and "
            "RSI. Returns a risk-on/neutral/risk-off classification, labeled markers, and a list "
            "of actionable insights. Use it to set the overall risk backdrop before sizing trades."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_VOLUME = {
    "type": "function",
    "function": {
        "name": "analyze_volume",
        "description": (
            "Analyze recent trade volume to gauge participation. Two gauges that answer "
            "different questions: `rvol_pace` is today's cumulative volume vs what an "
            "average day has accumulated by this same minute (1.0 = normal pace, 2.0+ = "
            "clearly elevated), while the local `relative_volume` (last 10 bars vs the "
            "prior 10) is participation RIGHT NOW. Pace stays elevated all session once a "
            "name is busy, so it alone cannot tell a live move from one buyers have already "
            "left -- use `participation_ok`, which is true only when BOTH clear (pace >= 2.0 "
            "and local >= 1.2), and `volume_burst` (last 3 bars vs the 10-bar average) for "
            "the shortest-horizon read. Also returns the on-balance-volume trend and whether "
            "volume confirms or diverges from the recent price move. Volume is sourced from "
            "the consolidated tape (yfinance, all exchanges) for accuracy rather than "
            "Alpaca's single-venue feed. IMPORTANT: armed tactic conditions on rvol_pace "
            "are evaluated against the trading feed's own counter, returned here as "
            "`rvol_pace_armable` -- when arming a pace condition, set the threshold "
            "against `rvol_pace_armable`, not `rvol_pace`. Labeled values plus a "
            "one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_CONSOLIDATION = {
    "type": "function",
    "function": {
        "name": "analyze_consolidation",
        "description": (
            "Measure the most recent consolidation/base in the intraday bars -- the flag of a "
            "bull-flag setup. Returns the base's high (`base_high`, the objective breakout "
            "trigger to arm a buy at), its low (`base_low`, the structural stop), its height "
            "(for a measured-move target), whether the range has contracted on declining volume "
            "(`is_coiling` -- a genuine tight flag), and how many times each edge has been "
            "tested. Use these measured levels instead of estimating the flag high by eye."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "base_bars": {
                    "type": "integer",
                    "description": (
                        "Number of most recent bars treated as the candidate base "
                        "(default 10, max 60)."
                    ),
                }
            },
            "required": [],
        },
    },
}

_TOOL_GET_KEY_LEVELS = {
    "type": "function",
    "function": {
        "name": "get_key_levels",
        "description": (
            "Map the session's structural support/resistance levels around the current price: "
            "prior-day high/low/close, premarket high/low, opening-range high/low, and the "
            "session high/low so far. Returns every level with its distance from spot, plus the "
            "nearest overhead resistance (the realistic first target and 'room to run' cap for "
            "a long entry) and the nearest support below (the stop anchor). An empty overhead "
            "list means blue-sky territory -- no structural resistance above."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# --- Advanced level tools (steps 4-6 of the S/R plan): implemented and
# dispatch-wired, but NOT yet exposed to any personality -- except
# analyze_volume_profile_2, which the Volume Signal Detective is built around
# (see VOLUME_DETECTIVE_TOOLS). To enable the others for the Momentum Trader,
# uncomment the three entries in MOMENTUM_TOOLS and the MOMENTUM_SYSTEM_PROMPT
# reassignment under MOMENTUM_ADVANCED_LEVELS_ADDENDUM.

_TOOL_ANALYZE_SWING_LEVELS = {
    "type": "function",
    "function": {
        "name": "analyze_swing_levels",
        "description": (
            "Locate clustered swing-point (fractal) support/resistance in the intraday bars: "
            "confirmed local highs/lows, merged within ~0.25 ATR and ranked by how many times "
            "each level was tested and how recently. A level tested 3+ times is far stronger "
            "than any single extreme print. Returns the ranked clusters plus the nearest swing "
            "resistance above and support below the current price."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "swing": {
                    "type": "integer",
                    "description": (
                        "Bars required on each side to confirm a swing point (default 3, max 10); "
                        "larger means fewer, more significant levels."
                    ),
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_VOLUME_PROFILE = {
    "type": "function",
    "function": {
        "name": "analyze_volume_profile",
        "description": (
            "Build a volume-by-price profile of the intraday bars: the Point of Control (the "
            "price with the most transacted volume -- a magnet/defended level), the 70% value "
            "area, high-volume nodes (support/resistance where positions were built), and "
            "low-volume nodes (air pockets price travels through quickly). An LVN just above "
            "the entry with the next HVN well higher improves the realistic first target."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bins": {
                    "type": "integer",
                    "description": "Number of price slices in the profile (default 24, max 60).",
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Optional trading day to profile, as 'YYYY-MM-DD'. Omit for today's live "
                        "intraday bars. Pass a prior trading date (e.g. the previous session) to "
                        "build the profile from that day's intraday bars instead -- useful for "
                        "comparing today's price against yesterday's POC and value area. Must be an "
                        "actual trading day within roughly the last 60 days; weekends/holidays "
                        "return no data."
                    ),
                },
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_VOLUME_PROFILE_2 = {
    "type": "function",
    "function": {
        "name": "analyze_volume_profile_2",
        "description": (
            "Recent support/resistance from one session's minute-by-minute volume "
            "(yfinance consolidated tape). Finds intraday volume spikes -- local "
            "surges past the opening warm-up -- and classifies each by how price "
            "momentum shifted across it: up-into-flat/down is a SUPPLY line "
            "(distribution/resistance), down-into-flat/up is a DEMAND line "
            "(accumulation/support), a catalyst printing near it flags it news-driven, "
            "else unsure. Then rebuilds a volume-by-price histogram with the spike "
            "volume removed to surface SCATTERED levels (size built up gradually, not "
            "in one burst). Any levels that formed before the most recent news-driven "
            "spike are dropped as stale. Returns the surviving peaks "
            "[{price, time, date, rel_vol, vol, type}] plus nearest support/resistance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bins": {
                    "type": "integer",
                    "description": "Price slices for the scattered-level histogram (default 24, max 60).",
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Optional trading day to analyze, as 'YYYY-MM-DD'. Omit for today's "
                        "live session. A prior date pulls that day's 1-min bars and news from "
                        "history; must be an actual trading day within roughly the last 60 days."
                    ),
                },
            },
            "required": [],
        },
    },
}

_TOOL_GET_FLOOR_PIVOTS = {
    "type": "function",
    "function": {
        "name": "get_floor_pivots",
        "description": (
            "Compute classic floor-trader pivot levels (P, R1-R3, S1-S3) from the prior "
            "completed session's high/low/close. Formula levels rather than structure, but "
            "watched widely enough to act as intraday reaction points; a pivot coinciding with "
            "a structural level (session high, swing cluster, high-volume node) is reinforced. "
            "Returns the levels split around the current price, nearest first."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_OPENING_RANGE = {
    "type": "function",
    "function": {
        "name": "analyze_opening_range",
        "description": (
            "Analyze today's Opening Range Breakout (ORB) setup: the high/low printed in "
            "the 09:30 ET + N minutes window of today's session (measured from bar "
            "timestamps, recovered via a targeted history fetch when the live buffer "
            "doesn't reach back to the open), whether price has since broken out above or "
            "below that range, and whether recent volume confirms the breakout. When the "
            "range genuinely cannot be established the result carries only a `note` -- "
            "there is no valid ORB setup in that case, do not improvise one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {
                    "type": "integer",
                    "description": "Length of the opening range in minutes (default 15).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_DETECT_REGIME_SHIFT = {
    "type": "function",
    "function": {
        "name": "detect_regime_shift",
        "description": (
            "Detect whether the ticker's intraday TRAJECTORY has turned -- the read a "
            "support/resistance map cannot give you, since levels describe where price "
            "WAS defended. Segments today's session into momentum legs and reports: the "
            "leg in force now versus the one it replaced and the `turn` between them "
            "(reversal / stall / resumption, with its price and how many minutes ago), "
            "`giveback_pct_of_prior_leg` (how much of the previous leg has been retraced "
            "-- 50%+ is a reversal, not a pause), `velocity` right now (accelerating, "
            "decelerating, or already reversing) as the earliest warning, the session "
            "VWAP with the last cross of it, `structure_break` (the last confirmed swing "
            "low broken to the downside, or swing high to the upside), `turn_volume_rel` "
            "(participation behind the turn vs the session's median minute), and whether "
            "news landed at the turn. Returns `shift_detected`, a `regime` label, a "
            "long-side `bias`, `levels_stale` (re-run the level map -- the levels behind "
            "any armed plan predate this turn), `reference_levels` to arm tactics on, "
            "insights, and a one-line summary. Uses the live streamed bars, so it reacts "
            "on the current tape rather than a delayed feed."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_SESSION_CLOCK = {
    "type": "function",
    "function": {
        "name": "get_session_clock",
        "description": (
            "Classify the current point in the trading day for breakout timing "
            "discipline: opening window (first 90 min -- historically the most reliable "
            "for breakouts), midday dead zone (12:00-14:00 ET -- notoriously fakeout-"
            "prone), power hour (final hour -- favorable), or outside regular hours. "
            "Returns the ET time, the window label, and `favorable_for_breakouts`. "
            "Check it before arming any breakout entry; in an unfavorable window demand "
            "much stronger confirmation or stand aside."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_PUT_CALL_WALLS = {
    "type": "function",
    "function": {
        "name": "get_put_call_walls",
        "description": (
            "Read the Call Wall (resistance, from peak call open interest) and Put Wall "
            "(support, from peak put open interest) for the ticker, plus the net dealer-gamma "
            "regime (positive = dampening, negative = amplifying) and whether those walls have "
            "been rising/falling recently. Uses the options chain most recently fetched in the "
            "background -- does not fetch fresh data itself. Returns labeled values, actionable "
            "insights, and a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_NEWS = {
    "type": "function",
    "function": {
        "name": "get_news",
        "description": "Get recent news headlines/summaries for the ticker, with impact labels where available.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max number of articles (default 10)."}},
            "required": [],
        },
    },
}

_TOOL_GET_POSITION = {
    "type": "function",
    "function": {
        "name": "get_position",
        "description": (
            "Get the current paper trading position size, cash balance, and total "
            "portfolio value (cash + position marked to the latest price)."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# Bullet list of every watchable field, injected into the alert tool description
# so the model always sees the current, authoritative set.
_ALERT_FIELDS_DOC = "; ".join(f"'{name}' ({desc})" for name, desc in ALERTABLE_FIELDS.items())

_TOOL_SUBMIT_DECISION = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": (
            "Finalize this trading cycle with exactly one decision: buy, sell, or "
            "alert. Must be called exactly once, after analysis is complete. When you "
            "don't want to trade, use 'alert' -- there is no do-nothing action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["buy", "sell", "alert"]},
                "symbol": {
                    "type": "string",
                    "description": (
                        "Ticker to trade -- required for buy/sell (must be one of your "
                        "streamed tickers). Ignored for alert: each alert entry names "
                        "its own symbol."
                    ),
                },
                "quantity": {
                    "type": "integer",
                    "description": (
                        "Whole shares to buy/sell (an integer; fractions are rounded down). "
                        "Ignored for alert. Must be at least 1 for buy/sell."
                    ),
                },
                "regime": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"],
                    "description": "The trading regime established during analysis.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Concise justification covering regime, strategy, and why this action follows from it.",
                },
                "alerts": {
                    "type": "array",
                    "description": (
                        "When action is 'alert': one or more conditions on continuously-updated "
                        "live data that should wake you early -- the instant any one is met -- "
                        "instead of sleeping out the full cycle. Each condition watches one "
                        "field, with 'above' meaning the field reaches or exceeds the value and "
                        "'below' meaning it reaches or falls to the value. Provide several to "
                        "watch a range or multiple signals at once; the first to trigger wakes "
                        f"you. Watchable fields: {_ALERT_FIELDS_DOC}."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Ticker whose live field to watch -- one of your streamed tickers.",
                            },
                            "field": {
                                "type": "string",
                                "enum": list(ALERTABLE_FIELDS.keys()),
                                "description": "Which live state field to watch.",
                            },
                            "condition": {
                                "type": "string",
                                "enum": ["above", "below"],
                                "description": "'above' = field >= value; 'below' = field <= value.",
                            },
                            "value": {
                                "type": "number",
                                "description": "Threshold the field is compared against.",
                            },
                        },
                        "required": ["symbol", "field", "condition", "value"],
                    },
                },
            },
            "required": ["action", "reasoning"],
        },
    },
}

_TACTIC_FIELDS_DOC = "; ".join(f"'{name}' ({desc})" for name, desc in TACTIC_CONDITION_FIELDS.items())

_TOOL_SET_TACTICS = {
    "type": "function",
    "function": {
        "name": "set_tactics",
        "description": (
            "Arm a standing conditional trade plan, executed for you the moment its "
            "conditions are met -- the preferred way to act on concrete levels instead of "
            "trading at the current price or waiting on a bare alert. Each action is a "
            "buy/sell with a size and one or more conditions that must ALL hold "
            "simultaneously; the first action whose conditions are met executes through "
            "the normal paper-fill path, the remaining actions are disarmed, and you are "
            "woken immediately to reevaluate. Replaces any previously armed tactics "
            "(pass an empty actions array to cancel them). Call at most once per cycle, "
            "then still finalize with submit_decision -- with tactics armed, action "
            "'alert' may carry an empty alerts array. On an open long position, a sell "
            "stop (last_price below, armed under your entry price) paired with a sell "
            "take-profit (last_price above) is trailed up automatically as price "
            "advances toward the target; the take-profit level never moves. Buy "
            "actions never fill outside the regular session or in its final 15 "
            "minutes (sells stay live to the bell); give entries an expires_at when "
            "they should not outlive the read that justified them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": (
                        "Conditional actions, evaluated independently -- e.g. an entry, a "
                        "stop-loss, and a take-profit are three actions. Empty array cancels "
                        "all armed tactics."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["buy", "sell"]},
                            "quantity": {
                                "type": "integer",
                                "description": (
                                    "Whole shares to trade (an integer; fractions are rounded "
                                    "down). Provide exactly one of quantity or quantity_pct."
                                ),
                            },
                            "quantity_pct": {
                                "type": "number",
                                "description": (
                                    "Percent (0-100] resolved at execution time: of the current "
                                    "position for a sell, of available cash for a buy, rounded "
                                    "down to whole shares. E.g. sell 20% of shares, or buy with "
                                    "50% of cash."
                                ),
                            },
                            "conditions": {
                                "type": "array",
                                "description": (
                                    "Conditions that must ALL hold at the same moment for this "
                                    f"action to execute. Watchable fields: {_TACTIC_FIELDS_DOC}."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {
                                            "type": "string",
                                            "enum": list(TACTIC_CONDITION_FIELDS.keys()),
                                            "description": "Which live field to watch.",
                                        },
                                        "condition": {
                                            "type": "string",
                                            "enum": ["above", "below"],
                                            "description": "'above' = field >= value; 'below' = field <= value.",
                                        },
                                        "value": {
                                            "type": "number",
                                            "description": "Threshold the field is compared against.",
                                        },
                                        "hold_sec": {
                                            "type": "number",
                                            "description": (
                                                "Optional (0-600, default 0): seconds the comparison must hold "
                                                "CONTINUOUSLY before it counts as met. Use on entries to demand a "
                                                "sustained cross instead of firing on a single wick tick (protective "
                                                "stops should usually stay at 0 so they react instantly)."
                                            ),
                                        },
                                    },
                                    "required": ["field", "condition", "value"],
                                },
                            },
                            "note": {
                                "type": "string",
                                "description": "Short label for this leg, e.g. 'entry on retest', 'stop-loss', 'take profit'.",
                            },
                            "trail": {
                                "type": "boolean",
                                "description": (
                                    "Optional (default true): whether the automatic trailing-stop ratchet "
                                    "may raise this action's protective sell stop. Set false to pin a "
                                    "structure stop exactly where you armed it."
                                ),
                            },
                            "expires_at": {
                                "type": "string",
                                "description": (
                                    "Optional ISO-8601 timestamp after which this action is retired "
                                    "unfired (its siblings stay armed). Use it to keep a resting entry "
                                    "bid from waiting on a tape that has moved on -- e.g. expire an "
                                    "entry a few cycles out unless re-armed. Omit for no expiry."
                                ),
                            },
                        },
                        "required": ["action", "conditions"],
                    },
                },
                "reasoning": {
                    "type": "string",
                    "description": "Concise justification for the plan: the setup and the levels it encodes.",
                },
            },
            "required": ["actions", "reasoning"],
        },
    },
}

_TOOL_STAND_DOWN = {
    "type": "function",
    "function": {
        "name": "stand_down",
        "description": (
            "Relinquish control back to the Automatic orchestrator because the market "
            "regime no longer fits your strategy and your setup is unlikely to appear "
            "soon. Available only in Automatic mode. Use this INSTEAD of submit_decision "
            "to end the cycle when standing aside on an alert would just be idling in the "
            "wrong regime. Does not close open positions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Why your strategy's edge is absent now and unlikely to return soon -- "
                        "cite the regime read (trend/range/volatility/volume) that no longer fits."
                    ),
                },
                "expected_quiet_minutes": {
                    "type": "number",
                    "description": "Rough estimate of how long the drought for your setup is likely to last, in minutes.",
                },
            },
            "required": ["reasoning"],
        },
    },
}

_TOOL_ANALYZE_VWAP_BANDS = {
    "type": "function",
    "function": {
        "name": "analyze_vwap_bands",
        "description": (
            "Analyze today's session VWAP and its volume-weighted standard-deviation bands for "
            "a mean-reversion read: the VWAP, the 1/2/3-sigma bands, price's signed z-score "
            "(how many std devs it sits from VWAP), the ADX trend-strength reading and whether "
            "it confirms a range (below 20), and whether the latest bar is a rejection candle. "
            "The `signal` is 'long_setup' (oversold >= trigger sigma below VWAP in a confirmed "
            "range), 'short_setup' (overbought above VWAP), 'no_setup_trending' (stretched but "
            "ADX shows a trend -- do not fade), or 'no_setup'. Returns labeled values plus a "
            "one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "num_std": {
                    "type": "number",
                    "description": "Std-dev stretch that triggers a setup (default 2.0).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_VWAP_REVERSION_GEOMETRY = {
    "type": "function",
    "function": {
        "name": "vwap_reversion_geometry",
        "description": (
            "Compute the mechanical entry/stop/target math for a VWAP mean-reversion trade. "
            "Target is always VWAP; the stop sits one standard deviation beyond entry (past the "
            "next band). Returns the reward-to-risk ratio and whether it clears the 1.5:1 "
            "mean-reversion minimum. Use this instead of doing the arithmetic yourself before "
            "sizing a trade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {"type": "number", "description": "Planned entry price (at/near the band)."},
                "vwap": {"type": "number", "description": "Session VWAP -- the reversion target."},
                "std_dev": {
                    "type": "number",
                    "description": "One standard deviation (the 1σ value from analyze_vwap_bands).",
                },
                "side": {
                    "type": "string",
                    "enum": ["long", "short"],
                    "description": "'long' for a stretch below VWAP, 'short' for a stretch above.",
                },
            },
            "required": ["entry", "vwap", "std_dev"],
        },
    },
}

_TOOL_BREAKOUT_TRADE_GEOMETRY = {
    "type": "function",
    "function": {
        "name": "breakout_trade_geometry",
        "description": (
            "Compute the mechanical entry/stop/target math for a long breakout trade: targets "
            "projected from ATR and/or the base height (1x and 2x each), the resulting "
            "reward-to-risk ratio for each, and whether the best one clears the 2:1 minimum. "
            "Pass the nearest overhead resistance (from get_key_levels) to also get "
            "`room_to_run` -- whether that ceiling sits at least 2x the stop distance above "
            "the entry. Use this instead of doing the arithmetic yourself before sizing a trade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {"type": "number", "description": "Planned entry price."},
                "stop": {"type": "number", "description": "Planned stop-loss price, below entry."},
                "atr": {
                    "type": "number",
                    "description": "ATR (from analyze_intraday_momentum), for an ATR-multiple target.",
                },
                "base_height": {
                    "type": "number",
                    "description": (
                        "Height of the consolidation base (from analyze_consolidation), for a "
                        "measured-move target."
                    ),
                },
                "overhead_resistance": {
                    "type": "number",
                    "description": (
                        "Nearest structural resistance level above the entry (from "
                        "get_key_levels), to check whether the trade has room to run before "
                        "hitting a ceiling."
                    ),
                },
            },
            "required": ["entry", "stop"],
        },
    },
}

_TOOL_ANALYZE_ORDER_BLOCKS = {
    "type": "function",
    "function": {
        "name": "analyze_order_blocks",
        "description": (
            "Locate institutional order blocks on the daily (higher) timeframe: bullish demand "
            "zones (the last down candle before an up-move that broke structure) and bearish supply "
            "zones (the mirror). Returns every block with its high/low boundaries, whether it has "
            "been mitigated (already revisited), and how many bars ago it formed, plus the nearest "
            "bullish demand block at/below price (a candidate entry on a return) and the nearest "
            "bearish supply block above (a candidate target). Labeled values plus a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_FAIR_VALUE_GAPS = {
    "type": "function",
    "function": {
        "name": "analyze_fair_value_gaps",
        "description": (
            "Locate fair value gaps (FVGs) -- three-candle price imbalances -- in recent intraday "
            "bars. Returns each gap's boundaries and whether it has been filled, plus the nearest "
            "bullish FVG at/below price (a support imbalance price may be filling now). A held fill "
            "of a bullish FVG is one of the intraday confirmations for a Smart Money long entry. "
            "Labeled values plus a one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent intraday bars to scan (default 50, max 300).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_SMART_MONEY_SETUP = {
    "type": "function",
    "function": {
        "name": "analyze_smart_money_setup",
        "description": (
            "The composite Smart Money read: ties a higher-timeframe bullish demand order block "
            "(daily) to today's intraday price action. Returns the demand block being watched, "
            "whether price is inside it, which intraday confirmations are present (bullish "
            "rejection candle, filled bullish FVG, or intraday break-of-structure/breaker), the "
            "suggested entry/stop (just beyond the block)/structural target, the reward-to-risk to "
            "that target, and a `signal`: 'long_setup' (return into demand, confirmed, clears 3:1), "
            "'watching' (valid block but not all conditions met), or 'no_setup'. Plus a `quality` "
            "grade and a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_SMART_MONEY_GEOMETRY = {
    "type": "function",
    "function": {
        "name": "smart_money_trade_geometry",
        "description": (
            "Compute the mechanical entry/stop/target math for a long Smart Money setup: entry at "
            "the demand block on a return, stop just beyond the block, target at the next opposing "
            "structural level. Returns the reward-to-risk ratio and whether it clears the 3:1 "
            "minimum this setup demands (it typically runs 3:1 to 5:1). Use this instead of doing "
            "the arithmetic yourself before sizing a trade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {"type": "number", "description": "Planned entry price, inside the order block."},
                "stop": {"type": "number", "description": "Planned stop-loss price, just beyond (below) the block."},
                "target": {"type": "number", "description": "Target price -- the next opposing structural level above entry."},
            },
            "required": ["entry", "stop", "target"],
        },
    },
}

_TOOL_ANALYZE_LIQUIDITY = {
    "type": "function",
    "function": {
        "name": "analyze_liquidity",
        "description": (
            "Map resting liquidity and recent stop-runs on the intraday timeframe -- the core "
            "Smart Money 'stop hunt' read. Returns buy-side liquidity pools above price (clustered "
            "swing highs where buy stops rest) and sell-side pools below (swing lows where sell "
            "stops rest), the nearest of each to price, and whether a recent liquidity SWEEP "
            "occurred: price piercing a prior swing level then closing back through it. A bullish "
            "sweep (a swing low undercut and reclaimed -- a stop-run below support that reversed) "
            "is one of the strongest intraday confirmations for a long off a demand block. Labeled "
            "values plus a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_PREMIUM_DISCOUNT = {
    "type": "function",
    "function": {
        "name": "analyze_premium_discount",
        "description": (
            "Locate price within the recent daily dealing range: the range high/low, its midpoint "
            "(equilibrium), and whether price sits in the DISCOUNT half (cheap, below equilibrium -- "
            "where Smart Money buys), the PREMIUM half (expensive, above it -- where Smart Money "
            "sells), or at equilibrium. Also returns the deep-discount OTE (optimal trade entry) "
            "zone, the 0.618-0.79 retracement down from the high. A long off a demand block that is "
            "ALSO in discount is higher quality than the same block in premium. Labeled values plus "
            "a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_SMART_MONEY_FLOW = {
    "type": "function",
    "function": {
        "name": "get_smart_money_flow",
        "description": (
            "The institutional 'smart money' ownership footprint for the ticker, from free SEC-"
            "derived disclosures: the percentage of shares held by insiders vs institutions, net "
            "insider buying/selling over the trailing 6 months (Form 4), and the largest "
            "institutional holders with their quarter-over-quarter share changes (13F). This is "
            "slow-moving (quarterly/Form-4 cadence), not an intraday timing signal -- use it as "
            "corroboration: net insider/institutional ACCUMULATION behind a bullish demand block "
            "strengthens the long thesis; DISTRIBUTION is a caution flag. Labeled values plus a "
            "one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_ANALYST_TARGETS = {
    "type": "function",
    "function": {
        "name": "get_analyst_targets",
        "description": (
            "Current Wall Street price targets for the ticker with the actionable read: the "
            "yfinance CONSENSUS (mean/median/high/low target across every covering analyst, the "
            "analyst count, and the recommendation) plus the standing target from UBS, Morgan "
            "Stanley, and Barclays -- each annotated with the implied upside/downside vs the "
            "current price. Use it to gauge how much room the Street sees: price near or above "
            "the consensus mean means limited upside (don't chase a gap into it -- it acts as "
            "resistance/an objective); a wide gap below the mean leaves room to run; price "
            "outside the whole high-low range is a valuation extreme. Targets update at most a "
            "few times a day (cached), so this is positional context, not an intraday trigger. "
            "Returns labeled values, a list of actionable `insights`, and a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_CORPORATE_ACTIONS = {
    "type": "function",
    "function": {
        "name": "get_corporate_actions",
        "description": (
            "Incoming corporate actions scheduled for the ticker over the next two weeks "
            "(configurable): cash/stock dividends with their ex-dividend and payable dates, "
            "forward/reverse splits, mergers, spin-offs, and similar events, flattened into one "
            "chronological list. These are scheduled, mechanical catalysts: an ex-dividend date "
            "lowers the open by roughly the dividend (not a bearish signal), a split resets every "
            "price level, and merger terms can pin or reprice the tape -- check them before "
            "trusting a gap read or leaving tactics armed across an event date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "description": "Lookahead window in days (default 14, max 90).",
                },
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_PREMARKET = {
    "type": "function",
    "function": {
        "name": "analyze_premarket",
        "description": (
            "Pre-market read for the upcoming session: the previous close, the latest "
            "pre-market price and the implied opening gap percentage, the pre-market "
            "high/low/volume printed so far from the early bars, and how many minutes "
            "remain until the opening bell. Use it to estimate where the stock will "
            "open before deriving your buy/sell levels."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_SYMBOL_PARAM: dict = {
    "type": "string",
    "description": "Ticker symbol this call applies to -- one of your streamed tickers.",
}


def _add_symbol_param(tool: dict) -> dict:
    """Give a per-ticker tool a required `symbol` argument (in place)."""
    params = tool["function"]["parameters"]
    params["properties"] = {"symbol": copy.deepcopy(_SYMBOL_PARAM), **params.get("properties", {})}
    required = [r for r in (params.get("required") or []) if r != "symbol"]
    params["required"] = ["symbol", *required]
    return tool


# Every tool that reads one ticker's data takes a required `symbol`. The
# exceptions are basket-wide reads (get_position, analyze_market), the pure
# geometry calculators, and the terminal decision tools (which carry symbols
# in their own payloads).
for _tool in (
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_CONSOLIDATION,
    _TOOL_GET_KEY_LEVELS,
    _TOOL_ANALYZE_SWING_LEVELS,
    _TOOL_ANALYZE_VOLUME_PROFILE,
    _TOOL_ANALYZE_VOLUME_PROFILE_2,
    _TOOL_DETECT_REGIME_SHIFT,
    _TOOL_GET_FLOOR_PIVOTS,
    _TOOL_GET_PUT_CALL_WALLS,
    _TOOL_GET_NEWS,
    _TOOL_ANALYZE_VWAP_BANDS,
    _TOOL_ANALYZE_ORDER_BLOCKS,
    _TOOL_ANALYZE_FAIR_VALUE_GAPS,
    _TOOL_ANALYZE_SMART_MONEY_SETUP,
    _TOOL_ANALYZE_LIQUIDITY,
    _TOOL_ANALYZE_PREMIUM_DISCOUNT,
    _TOOL_GET_SMART_MONEY_FLOW,
    _TOOL_GET_ANALYST_TARGETS,
    _TOOL_GET_CORPORATE_ACTIONS,
    _TOOL_ANALYZE_PREMARKET,
    _TOOL_SET_TACTICS,
):
    _add_symbol_param(_tool)


# Momentum trader: RVOL + price action/VWAP + news + price, plus measured levels
# (consolidation base, session-structure S/R) and the R:R geometry check so
# entries anchor to data-based levels -- but still no medium-term regime or
# broad-market backdrop, the whole point is reacting fast to what's happening now.
MOMENTUM_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_CONSOLIDATION,
    _TOOL_GET_KEY_LEVELS,
    _TOOL_BREAKOUT_TRADE_GEOMETRY,
    # Advanced level sources (swing clusters, volume profile, floor pivots).
    # Enabled together with MOMENTUM_ADVANCED_LEVELS_ADDENDUM: step 3 asks for
    # the first level leaving 1.5 ATR of room rather than the nearest tick of
    # session structure, and touch-ranked swing clusters plus volume-profile
    # nodes are what make that level findable.
    _TOOL_ANALYZE_SWING_LEVELS,
    _TOOL_ANALYZE_VOLUME_PROFILE,
    _TOOL_GET_FLOOR_PIVOTS,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# Breakout trader: ORB + session clock discipline + volume + structural levels
# (room-to-run for breakout_trade_geometry) + broad-market backdrop + ATR-based
# targets + news + price.
BREAKOUT_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_GET_SESSION_CLOCK,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_GET_KEY_LEVELS,
    _TOOL_ANALYZE_MARKET,
    _TOOL_BREAKOUT_TRADE_GEOMETRY,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# VWAP mean-reversion trader: session VWAP bands + ADX regime gate + volume
# (exhaustion vs breakout) + reversion geometry + news + price. No daily trend
# or options positioning -- this is a fast intraday, regime-gated fade.
REVERSAL_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_VWAP_BANDS,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_VWAP_REVERSION_GEOMETRY,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# Smart Money (highest-edge): higher-timeframe daily structure (trend + order
# blocks) + intraday confirmation (the composite read + FVG drill-in) + volume +
# SMC geometry + news + price. The composite tool does the heavy lifting; the
# order-block / FVG tools let the agent drill into the structure behind it.
SMART_MONEY_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_ANALYZE_ORDER_BLOCKS,
    _TOOL_ANALYZE_PREMIUM_DISCOUNT,
    _TOOL_ANALYZE_SMART_MONEY_SETUP,
    _TOOL_ANALYZE_FAIR_VALUE_GAPS,
    _TOOL_ANALYZE_LIQUIDITY,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_GET_SMART_MONEY_FLOW,
    _TOOL_GET_ANALYST_TARGETS,
    _TOOL_SMART_MONEY_GEOMETRY,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# Volume Signal Detective: spike-classified demand/supply lines are the primary
# read (analyze_volume_profile_2), cross-examined against session structure
# (get_key_levels), participation (analyze_volume), the approach into the level
# (analyze_intraday_momentum: ATR + momentum), timing (get_session_clock), and
# the catalyst (get_news) -- with the 2:1 / room-to-run geometry gate before any
# size. detect_regime_shift is the counterweight to all of that: levels describe
# the past, and a level map alone will keep bidding demand into a tape that has
# already turned, so the trajectory read gates every plan the levels suggest.
# No daily trend or options positioning: the edge is one session's tape.
VOLUME_DETECTIVE_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_VOLUME_PROFILE_2,
    _TOOL_DETECT_REGIME_SHIFT,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_GET_KEY_LEVELS,
    _TOOL_GET_SESSION_CLOCK,
    _TOOL_BREAKOUT_TRADE_GEOMETRY,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# Premarket analyst: the pre-open read (gap, pre-market range, time to bell) +
# the catalyst + the daily structure and broad backdrop the open will trade
# against. No intraday tools -- there is no session yet; the whole output is a
# set_tactics bracket for the opening prints.
PREMARKET_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_PREMARKET,
    _TOOL_GET_NEWS,
    _TOOL_GET_CORPORATE_ACTIONS,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_GET_ANALYST_TARGETS,
    _TOOL_ANALYZE_MARKET,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

PERSONALITY_TOOLS: dict[str, list[dict]] = {
    "momentum": MOMENTUM_TOOLS,
    "breakout": BREAKOUT_TOOLS,
    "reversal": REVERSAL_TOOLS,
    "smart_money": SMART_MONEY_TOOLS,
    "volume_detective": VOLUME_DETECTIVE_TOOLS,
    "premarket": PREMARKET_TOOLS,
}


