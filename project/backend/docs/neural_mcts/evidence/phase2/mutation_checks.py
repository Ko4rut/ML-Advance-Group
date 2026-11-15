import sys, unittest
from unittest import mock
sys.path.insert(0, '.')
import src.encoding.observations as O, src.encoding.actions as A
from src.environment import Move

def run():
    suite = unittest.defaultTestLoader.discover('tests/encoding', top_level_dir='.')
    r = unittest.TextTestRunner(stream=open('/dev/null','w')).run(suite)
    return len(r.failures)+len(r.errors)

orig_plane = O.piece_plane
orig_to = A.CCLiteVocabulary.to_native
orig_from = A.CCLiteVocabulary.from_native
muts = {
 "baseline (no mutation)": [],
 "colors swapped (black planes first)": [mock.patch.object(O, "piece_plane", lambda p: abs(p)-1+(0 if p<0 else 7))],
 "piece order wrong (rook<->cannon)": [mock.patch.object(O, "piece_plane", lambda p: orig_plane(p if abs(p) not in (5,6) else (11-abs(p))*(1 if p>0 else -1)))],
 "turn plane inverted": [mock.patch.object(O, "PLANE_RED_TO_MOVE", 15), mock.patch.object(O, "PLANE_SIDE_IN_CHECK", 14)],
 "vocabulary transposed (target*90+source)": [mock.patch.object(A.CCLiteVocabulary, "to_native", lambda self,m: m.target*90+m.source)],
 "encoder transposed (target*90+source)": [mock.patch.object(A.ActionEncoder, "encode", lambda self,m: m.target*90+m.source)],
 "check plane never set": [mock.patch.object(O.RuleValidator, "is_in_check", lambda *a, **k: False)],
}
for name, patches in muts.items():
    for p in patches: p.start()
    try: n = run()
    finally:
        for p in patches: p.stop()
    print(f"{'DETECTED' if n else 'MISSED  '} ({n} failing) {name}" if patches else f"baseline failing={n}")
