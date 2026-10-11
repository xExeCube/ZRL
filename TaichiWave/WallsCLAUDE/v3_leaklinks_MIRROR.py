# v3 (01/10/2026): structural leak check with an own link builder: count free-free links that join
# side A to side B of a one-cell wall (a leak is possible iff this count > 0).
import numpy as np
offs6 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
ST = {'sq5': [((1,0),[]),((-1,0),[]),((0,1),[]),((0,-1),[])],
      'sq9': [((1,0),[]),((-1,0),[]),((0,1),[]),((0,-1),[])] + [((a,b),[(a,0),(0,b)]) for a in (1,-1) for b in (1,-1)],
      'tri': [(d, [c for c in offs6 if (d[0]-c[0], d[1]-c[1]) in offs6]) for d in offs6]}
n = 41
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
cases = {'straight': (I == 20, I > 20), 'diag i=j': (I == J, I < J), 'anti i+j=n-1': (I + J == n-1, I + J > n-1)}
for lat in ST:
    for cname, (wall, B) in cases.items():
        res = []
        for block in (False, True):
            cnt = 0
            for i in range(n):
                for j in range(n):
                    if wall[i, j] or B[i, j]: continue          # i on side A, free
                    for (di, dj), com in ST[lat]:
                        p, q = i+di, j+dj
                        if not (0 <= p < n and 0 <= q < n) or wall[p, q] or not B[p, q]: continue
                        if block and com and all(0 <= i+a < n and 0 <= j+b < n and wall[i+a, j+b] for a, b in com):
                            continue
                        cnt += 1
            res.append(f'{"blocked" if block else "no-block"}: {cnt} crossing links')
        print(f'{lat:3} {cname:13} ' + '  '.join(res))
