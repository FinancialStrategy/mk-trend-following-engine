from MK_Yahoo_Session_Aware_Strict_v0083 import _agreement_diagnostics, _values_agree

fixtures = [
    [4.547914028167725, 4.547914028167725, 4.547914505004883, 4.547913074493408],
    [4.555893421173096, 4.555893421173096, 4.555892467498779, 4.555892467498779],
    [4.547914028167725, 4.547914028167725, 4.547914505004883, 4.547913074493408],
    [4.539935111999512, 4.539935111999512, 4.53993558883667, 4.53993558883667],
    [4.571850776672363, 4.571850776672363, 4.571850299835205, 4.571850776672363],
    [4.699511528015137, 4.699511528015137, 4.6995110511779785, 4.69951057434082],
    [4.651638031005859, 4.651638031005859, 4.651638507843018, 4.651638507843018],
]

for i, values in enumerate(fixtures, 1):
    d = _agreement_diagnostics(values, "Adj Close")
    assert d["agree"], (i, d)
    assert d["spread_bps"] < 0.01, (i, d)

# Genuine economic disagreement must still stop.
bad = [4.547914028167725, 4.547914028167725, 4.56, 4.547914028167725]
d = _agreement_diagnostics(bad, "Adj Close")
assert not d["agree"]
assert d["spread_bps"] > 1.0

# Volume remains integer-strict.
assert _values_agree([10_000_000, 10_000_000, 10_000_000], "Volume")
assert not _values_agree([10_000_000, 10_000_002], "Volume")

print("PASS — Yahoo precision-aware consensus v0.08.3")
for i, values in enumerate(fixtures, 1):
    d = _agreement_diagnostics(values, "Adj Close")
    print(
        f"AKBNK fixture {i}: PASS | abs={d['absolute_spread']:.12g} "
        f"| {d['spread_bps']:.6f} bps | tol={d['tolerance']:.12g}"
    )
print("Material Adj Close conflict hard-stop: PASS")
print("Volume strict reconciliation: PASS")
print("Consensus averaging / median synthesis: NO")
print("Accepted value: highest-priority actually observed Yahoo value")
