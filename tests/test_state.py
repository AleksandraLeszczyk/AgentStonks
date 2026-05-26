import threading

from marketview.config import MAX_BARS
from marketview.state import AppState


def test_initial_values():
    s = AppState()
    assert list(s.bars) == []
    assert s.trades == []
    assert s.news == []
    assert s.symbol == ""
    assert s.status == "Idle"
    assert s.ws is None
    assert s.ws_news is None


def test_bars_deque_respects_max_bars():
    s = AppState()
    for i in range(MAX_BARS + 50):
        s.bars.append({"t": i})
    assert len(s.bars) == MAX_BARS


def test_lock_is_reentrant_from_multiple_threads():
    s = AppState()
    errors = []

    def writer(value):
        try:
            with s.lock:
                s.bars.append({"v": value})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(s.bars) == 50
