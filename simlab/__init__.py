"""SimLab — the strategy testing / simulation suite for AgentStonks.

A separate app (``streamlit run sim_main.py``) that replays the LLM trading
agents against stored historical minute bars instead of the live tape:

- ``data``     — dataset download + smart local storage (bars, news, market)
- ``engine``   — the simulation loop: real agent cycles at a pinned clock,
                 deterministic bar-by-bar fast-forward between them
- ``patches``  — the simulation context that reroutes live-fetch call sites
- ``judge``    — LLM-as-judge scoring of entries and the whole run
- ``results``  — run records, profit/oracle scores, persistence, Langfuse
- ``prompts``  — editable per-personality prompt overrides
- ``app``      — the three-tab Streamlit UI
"""
