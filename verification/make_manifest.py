"""Write MANIFEST.sha256: SHA-256 of every file in results/ and weights/, and of the pre-registration originals.
Run only when the stored results are deliberately regenerated:  python verification/make_manifest.py"""
import os, hashlib, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()
files = []
for top in ['results', 'weights']:
    for dp, _, fs in os.walk(os.path.join(ROOT, top)):
        files += [os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, '/') for f in fs]
files += sorted(os.path.relpath(p, ROOT).replace(os.sep, '/') for p in glob.glob(os.path.join(ROOT, 'preregistration', '*_事前登録.md')))
with open(os.path.join(ROOT, 'MANIFEST.sha256'), 'w', encoding='utf-8') as fh:
    for p in sorted(set(files)): fh.write(f'{sha256(os.path.join(ROOT, p))}  {p}\n')
print(len(set(files)), 'files written to MANIFEST.sha256')
