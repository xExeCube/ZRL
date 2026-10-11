"""Scratch: js_num_str vs Node's String(x) on awkward numbers (30/09/2026)."""
import json, os, random, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
import alignment_MIRROR as al
random.seed(1)
xs = [0.0, 1.0, -2.0, 1e-7, 1.5e-7, 9.99e-7, 1e-6, 1.234e-5, 1e-5, 0.0001, 0.1, 0.3, 16.75, 33.333333333333336,
      1e21, 1.5e21, 1e20, 123456789012345680000.0, 5e-324, 1.7976931348623157e308, -3.5e-9, 22.500000000000004]
xs += [random.uniform(-1, 1) * 10 ** random.randint(-12, 25) for _ in range(3000)]
js = subprocess.run(['node', '-e', 'const a=JSON.parse(require("fs").readFileSync(0,"utf8"));'
                     'console.log(JSON.stringify(a.map(x=>String(x))))'],
                    input=json.dumps(xs), capture_output=True, text=True, check=True)
J = json.loads(js.stdout)
bad = [(x, j, al.js_num_str(x)) for x, j in zip(xs, J) if j != al.js_num_str(x)]
print(f'{len(xs) - len(bad)}/{len(xs)} agree'); print(bad[:10])
