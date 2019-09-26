import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np

file1 = './grui.txt'


df1 = pd.read_csv(file1, sep='\s+', header=0, index_col=False)


df4 = df1[(abs(df1.gsharp) < 2.5) & (abs(df1.rsharp) < 2.5)
          & (df1.gchi < 1.) & (df1.rchi < 1.2)]
gp = np.polyfit(df4.rmag, np.log(df4.rmerr), 1)
rp = np.polyfit(df4.gmag, np.log(df4.gmerr), 1)
df5 = df4[(np.log(df4.rmerr) < ((gp[1] + .3) + (df4.rmag * gp[0])))
          & (np.log(df4.gmerr) < ((rp[1] + .3) + (df4.gmag * rp[0])))]


x = np.linspace(15, 20, len(df4))
yp = np.full_like(x, 2)
ym = np.full_like(x, -2)

plt.figure(figsize=(10, 10))
plt.subplot2grid((2, 2), (0, 0), colspan=2)

plt.scatter(df5.rmag, df5.rsharp, edgecolors='none', s=5, alpha=.2)
plt.plot(x, yp, color='red')
plt.plot(x, ym, color='red')

plt.xlim(15, 19)
plt.ylim(-5, 5)
plt.ylabel('r sharp')
plt.xlabel('r (instrumental)')

plt.subplot2grid((2, 2), (1, 0))
plt.scatter(df5.rmag, df5.rchi, edgecolors='none', s=5, alpha=.2)
plt.xlim(15, 19)
plt.ylim(-0.1, 3)

plt.ylabel('r chi')
plt.xlabel('r (instrumental)')

plt.subplot2grid((2, 2), (1, 1))
plt.scatter(df5.rmag, np.log(df5.rmerr), edgecolors='none', s=5, alpha=.2)
plt.plot(x, gp[1] + x * gp[0], color='red')
plt.plot(x, (gp[1] + x * gp[0] - .3), color='red', label=('fit +/- .3'))
plt.plot(x, (gp[1] + x * gp[0] + .3), color='red')
plt.xlim(16, 19)
plt.ylim(-5, 0)
plt.legend(loc='best', fancybox=True)
plt.xlabel('r (instrumental)')
plt.ylabel('log(r err)')

plt.suptitle('Grus I')


plt.savefig('look.png', dpi=350, bbbox_inches='tight')


plt.figure(figsize=(10, 10))
plt.subplot2grid((2, 2), (0, 0), colspan=2)

plt.scatter(df5.gmag, df5.gsharp, edgecolors='none', s=5, alpha=.2)
plt.plot(x, yp, color='red')
plt.plot(x, ym, color='red')

plt.xlim(15, 19)
plt.ylim(-5, 5)
plt.ylabel('g sharp')
plt.xlabel('g (instrumental)')

plt.subplot2grid((2, 2), (1, 0))
plt.scatter(df5.gmag, df5.gchi, edgecolors='none', s=5, alpha=.2)
plt.xlim(15.5, 20)
plt.ylim(-0.1, 3)

plt.ylabel('g chi')
plt.xlabel('g (instrumental)')

plt.subplot2grid((2, 2), (1, 1))
plt.scatter(df5.gmag, np.log(df5.gmerr), edgecolors='none', s=5, alpha=.2)
plt.plot(x, rp[1] + x * rp[0], color='red')
plt.plot(x, (rp[1] + x * rp[0] - .3), color='red', label=('fit +/- .3'))
plt.plot(x, (rp[1] + x * rp[0] + .3), color='red')
plt.xlim(16, 19)
plt.ylim(-5, 0)
plt.legend(loc='best', fancybox=True)
plt.xlabel('g (instrumental)')
plt.ylabel('log(g err)')

plt.suptitle('Grus I')


plt.savefig('look1.png', dpi=350, bbbox_inches='tight')
