"""Test temporal del clamping de fetch_more_rows (dataset Spanish limitado)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import train_cli


class FakeDS:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, fn):
        return FakeDS([r for r in self._rows if fn(r)])

    def skip(self, n):
        return FakeDS(self._rows[n:])

    def __iter__(self):
        return iter(self._rows)


def make_stream(total=2600, spanish_every=5):
    # 2600 filas, 1 de cada 5 Spanish -> 520 Spanish en total
    return [{"language": "Spanish" if i % spanish_every == 0 else "English", "i": i}
            for i in range(total)]


# Caso 1: suficiente -> devuelve exactamente target (sin AVISO)
train_cli.load_dataset = lambda *a, **k: FakeDS(make_stream())
train_cli.existing_raw_rows = lambda: 400
train_cli.append_jsonl = lambda rws, p: None
train_cli.load_jsonl = lambda p: [{"_ok": True}] * (400 + len(train_cli._last))
train_cli._last = None
orig_append = train_cli.append_jsonl


def cap(rws, p):
    train_cli._last = rws


train_cli.append_jsonl = cap
train_cli.load_jsonl = lambda p: [{"_ok": True}] * 900  # 400 + las 500 disponibles
out = train_cli.fetch_more_rows(900)
assert len(out) == 900, len(out)
assert len(train_cli._last) == 500, len(train_cli._last)
print("Caso 1 (suficiente): OK ->", len(out), "filas")

# Caso 2: insuficiente -> clamp sin RuntimeError, con AVISO
# Spanish total = 520; con current=400 solo quedan 120; pedimos 900 -> clamp
train_cli.existing_raw_rows = lambda: 400
train_cli.load_jsonl = lambda p: [{"_ok": True}] * 520
out = train_cli.fetch_more_rows(2000)
assert len(out) == 520, len(out)
assert len(train_cli._last) == 120, len(train_cli._last)
print("Caso 2 (insuficiente): OK ->", len(out), "filas (clamp con AVISO)")

# Caso 3: cero restantes -> devuelve lo local sin error
train_cli.existing_raw_rows = lambda: 520
train_cli.load_jsonl = lambda p: [{"_ok": True}] * 520
out = train_cli.fetch_more_rows(600)
assert len(out) == 520, len(out)
print("Caso 3 (cero restantes): OK ->", len(out), "filas locales")

print("FETCH_MORE_ROWS OK")