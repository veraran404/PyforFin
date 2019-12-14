```python
from ImportAll import *
from Functions import *
```

# Chaper 9. Mathematical Tools

# 1.回归

## 1.1一元回归方程


```python
def f(x):
    return np.sin(x)+0.5*x
```


```python
x=np.linspace(-2*np.pi,2*np.pi,50)
plt.plot(x,f(x),'b')
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
#align_figures()
```


![png](/README/output_5_0.png)



```python
#一阶拟合
reg=np.polyfit(x,f(x),deg=1) #返回多项式系数行向量p(x) = p[0] * x**deg + ... + p[deg]
ry=np.polyval(reg,x) #计算Xi点多项式的值
```


```python
reg
```




    array([ 4.28841952e-01, -1.31499950e-16])




```python
ry
```




    array([-2.69449345, -2.58451412, -2.4745348 , -2.36455548, -2.25457615,
           -2.14459683, -2.0346175 , -1.92463818, -1.81465885, -1.70467953,
           -1.5947002 , -1.48472088, -1.37474156, -1.26476223, -1.15478291,
           -1.04480358, -0.93482426, -0.82484493, -0.71486561, -0.60488628,
           -0.49490696, -0.38492764, -0.27494831, -0.16496899, -0.05498966,
            0.05498966,  0.16496899,  0.27494831,  0.38492764,  0.49490696,
            0.60488628,  0.71486561,  0.82484493,  0.93482426,  1.04480358,
            1.15478291,  1.26476223,  1.37474156,  1.48472088,  1.5947002 ,
            1.70467953,  1.81465885,  1.92463818,  2.0346175 ,  2.14459683,
            2.25457615,  2.36455548,  2.4745348 ,  2.58451412,  2.69449345])




```python
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
#align_figures()
```


![png](output_9_0.png)



```python
#5阶拟合
reg = np.polyfit(x, f(x), deg=5)
ry = np.polyval(reg, x)
plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
#align_figures()
```


![png](output_10_0.png)



```python
#7阶拟合
reg = np.polyfit(x, f(x), deg=7)
ry = np.polyval(reg, x)
plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
#align_figures()
```


![png](output_11_0.png)



```python
#A brief check reveals that the result is not perfect:
np.allclose(f(x),ry)
```




    False




```python
#However, the mean squared error (MSE) is not too large:
np.sum((f(x)-ry)**2)/len(x)
```




    0.0017769134759517689



## 1.2多元回归方程


```python
matrix=np.zeros((3+1,len(x)))
matrix
```




    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.]])




```python
matrix[3,:]=x**3
matrix[2,:]=x**2
matrix[1,:]=x
matrix[0,:]=1
matrix
```




    array([[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             1.00000000e+00,  1.00000000e+00],
           [-6.28318531e+00, -6.02672876e+00, -5.77027222e+00,
            -5.51381568e+00, -5.25735913e+00, -5.00090259e+00,
            -4.74444605e+00, -4.48798951e+00, -4.23153296e+00,
            -3.97507642e+00, -3.71861988e+00, -3.46216333e+00,
            -3.20570679e+00, -2.94925025e+00, -2.69279370e+00,
            -2.43633716e+00, -2.17988062e+00, -1.92342407e+00,
            -1.66696753e+00, -1.41051099e+00, -1.15405444e+00,
            -8.97597901e-01, -6.41141358e-01, -3.84684815e-01,
            -1.28228272e-01,  1.28228272e-01,  3.84684815e-01,
             6.41141358e-01,  8.97597901e-01,  1.15405444e+00,
             1.41051099e+00,  1.66696753e+00,  1.92342407e+00,
             2.17988062e+00,  2.43633716e+00,  2.69279370e+00,
             2.94925025e+00,  3.20570679e+00,  3.46216333e+00,
             3.71861988e+00,  3.97507642e+00,  4.23153296e+00,
             4.48798951e+00,  4.74444605e+00,  5.00090259e+00,
             5.25735913e+00,  5.51381568e+00,  5.77027222e+00,
             6.02672876e+00,  6.28318531e+00],
           [ 3.94784176e+01,  3.63214596e+01,  3.32960415e+01,
             3.04021633e+01,  2.76398251e+01,  2.50090267e+01,
             2.25097683e+01,  2.01420498e+01,  1.79058712e+01,
             1.58012325e+01,  1.38281338e+01,  1.19865749e+01,
             1.02765560e+01,  8.69807701e+00,  7.25113793e+00,
             5.93573876e+00,  4.75187950e+00,  3.69956017e+00,
             2.77878075e+00,  1.98954125e+00,  1.33184166e+00,
             8.05681992e-01,  4.11062241e-01,  1.47982407e-01,
             1.64424896e-02,  1.64424896e-02,  1.47982407e-01,
             4.11062241e-01,  8.05681992e-01,  1.33184166e+00,
             1.98954125e+00,  2.77878075e+00,  3.69956017e+00,
             4.75187950e+00,  5.93573876e+00,  7.25113793e+00,
             8.69807701e+00,  1.02765560e+01,  1.19865749e+01,
             1.38281338e+01,  1.58012325e+01,  1.79058712e+01,
             2.01420498e+01,  2.25097683e+01,  2.50090267e+01,
             2.76398251e+01,  3.04021633e+01,  3.32960415e+01,
             3.63214596e+01,  3.94784176e+01],
           [-2.48050213e+02, -2.18899585e+02, -1.92127223e+02,
            -1.67631925e+02, -1.45312487e+02, -1.25067707e+02,
            -1.06796381e+02, -9.03973081e+01, -7.57692842e+01,
            -6.28111068e+01, -5.14215731e+01, -4.14994802e+01,
            -3.29436254e+01, -2.56528058e+01, -1.95258186e+01,
            -1.44614609e+01, -1.03585300e+01, -7.11582309e+00,
            -4.63213728e+00, -2.80626979e+00, -1.53701779e+00,
            -7.23178465e-01, -2.63549003e-01, -5.69265847e-02,
            -2.10839203e-03,  2.10839203e-03,  5.69265847e-02,
             2.63549003e-01,  7.23178465e-01,  1.53701779e+00,
             2.80626979e+00,  4.63213728e+00,  7.11582309e+00,
             1.03585300e+01,  1.44614609e+01,  1.95258186e+01,
             2.56528058e+01,  3.29436254e+01,  4.14994802e+01,
             5.14215731e+01,  6.28111068e+01,  7.57692842e+01,
             9.03973081e+01,  1.06796381e+02,  1.25067707e+02,
             1.45312487e+02,  1.67631925e+02,  1.92127223e+02,
             2.18899585e+02,  2.48050213e+02]])




```python
reg=np.linalg.lstsq(matrix.T,f(x),rcond=None)[0] #返回线性矩阵方程的最小二乘解，多元线性回归的矩阵形式
```


```python
reg #多元线性回归方程的系数
```




    array([ 1.50654604e-14,  5.62777448e-01, -1.11022302e-15, -5.43553615e-03])




```python
ry=np.dot(reg,matrix) #np.dot矩阵乘法,代入系数求回归方程的值
ry
```




    array([-2.18774909, -2.20187043, -2.20306461, -2.19188173, -2.16887188,
           -2.13458516, -2.08957165, -2.03438145, -1.96956464, -1.89567132,
           -1.81325159, -1.72285552, -1.62503322, -1.52033478, -1.40931028,
           -1.29250982, -1.17048349, -1.04378138, -0.91295358, -0.77855019,
           -0.6411213 , -0.50121699, -0.35938737, -0.21618251, -0.07215252,
            0.07215252,  0.21618251,  0.35938737,  0.50121699,  0.6411213 ,
            0.77855019,  0.91295358,  1.04378138,  1.17048349,  1.29250982,
            1.40931028,  1.52033478,  1.62503322,  1.72285552,  1.81325159,
            1.89567132,  1.96956464,  2.03438145,  2.08957165,  2.13458516,
            2.16887188,  2.19188173,  2.20306461,  2.20187043,  2.18774909])




```python
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
#align_figures()
```


![png](output_20_0.png)



```python
matrix[3,:]=np.sin(x)
reg=np.linalg.lstsq(matrix.T,f(x),rcond=None)[0]
ry=np.dot(reg,matrix)
reg
```




    array([4.2004068e-16, 5.0000000e-01, 0.0000000e+00, 1.0000000e+00])




```python
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
#align_figures()
```


![png](output_22_0.png)



```python
np.allclose(f(x),ry)
```




    True




```python
np.sum((f(x)-ry)**2)/len(x)
```




    3.404735992885531e-31



## 1.3噪音数据


```python
xn=np.linspace(-2*np.pi,2*np.pi,50)
xn=xn+0.5*np.random.standard_normal(len(xn))
yn=f(xn)+0.25*np.random.standard_normal(len(xn))
```


```python
reg=np.polyfit(xn,yn,7)
ry=np.polyval(reg,xn)
```


```python
plt.plot(xn,yn,'b^',label='f(x)')
plt.plot(xn,ry,'ro',label='regression')
plt.legend(loc=0)
plt.grid=True
plt.xlabel=('x')
plt.ylabel=('f(x)')
#align_figures()
```


![png](output_28_0.png)


## 1.4未排序数据


```python
xu=np.random.rand(50)*4*np.pi-2*np.pi
yu=f(xu)
```


```python
print(xu[:10].round(2))
print(yu[:10].round(2))
```

    [-5.13 -5.24 -5.45 -0.04  5.23 -4.68  0.1  -5.4   1.66 -3.23]
    [-1.65 -1.76 -1.99 -0.05  1.75 -1.34  0.15 -1.93  1.82 -1.53]



```python
reg=np.polyfit(xu,yu,5)
ry=np.polyval(reg,xu)
```


```python
plt.plot(xu, yu, 'b^', label='f(x)')
plt.plot(xu, ry, 'ro', label='regression')
plt.legend(loc=0)
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
#align_figures()
```


![png](output_33_0.png)


## 1.5多维回归


```python
def fm(x,y):
    return np.sin(x)+0.25*x+np.sqrt(y)+0.05*y**2
```


```python
x=np.linspace(0,10,20)
y=np.linspace(0,10,20)
X,Y=np.meshgrid(x,y) #generates 2-d grids out of the 1-d arrays
Z=fm(X,Y)
x=X.flatten() #yields 1-d arrays from the 2-d grids
y=Y.flatten()
```


```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
```


```python
fig=plt.figure(figsize=(9,6))
ax=fig.gca(projection='3d')
surf=ax.plot_surface(X,Y,Z,rstride=2,cstride=2,
                     cmap=mpl.cm.coolwarm,
                    linewidth=0.5,antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf,shrink=0.5,aspect=5)
#align_figures()
```




    <matplotlib.colorbar.Colorbar at 0x1c23703b00>




![png](output_38_1.png)



```python
matrix=np.zeros((len(x),6+1))
matrix[:,6]=np.sqrt(y)
matrix[:,5]=np.sin(x)
matrix[:,4]=y**2
matrix[:,3]=x**2
matrix[:,2]=y
matrix[:,1]=x
matrix[:,0]=1
```


```python
import statsmodels.api as sm
```


```python
model=sm.OLS(fm(x,y),matrix).fit()
```


```python
model.rsquared
```




    1.0




```python
a=model.params
a
```




    array([ 7.77156117e-15,  2.50000000e-01, -1.33226763e-15, -3.67761377e-16,
            5.00000000e-02,  1.00000000e+00,  1.00000000e+00])




```python
def reg_func(a,x,y):
    f6=a[6]*np.sqrt(y)
    f5=a[5]*np.sin(x)
    f4=a[4]*y**2
    f3=a[3]*x**2
    f2=a[2]*y
    f1=a[1]*x
    f0=a[0]*1
    return (f6+f5+f4+f3+f2+f1+f0)
#计算多维回归方程结果，无法用矩阵乘法np.dot
```


```python
RZ=reg_func(a,X,Y)
RZ[:10]
```




    array([[7.77156117e-15, 6.33930102e-01, 1.13188751e+00, 1.39470362e+00,
            1.38685613e+00, 1.14608395e+00, 7.73172323e-01, 4.04673032e-01,
            1.75943548e-01, 1.84509489e-01, 4.63667105e-01, 9.73470895e-01,
            1.61154576e+00, 2.24079713e+00, 2.72651873e+00, 2.97285383e+00,
            2.94874110e+00, 2.69632201e+00, 2.31953430e+00, 1.95597889e+00],
           [7.39326666e-01, 1.37325677e+00, 1.87121418e+00, 2.13403029e+00,
            2.12618279e+00, 1.88541061e+00, 1.51249899e+00, 1.14399970e+00,
            9.15270214e-01, 9.23836155e-01, 1.20299377e+00, 1.71279756e+00,
            2.35087242e+00, 2.98012380e+00, 3.46584539e+00, 3.71218050e+00,
            3.68806777e+00, 3.43564867e+00, 3.05886096e+00, 2.69530555e+00],
           [1.08138001e+00, 1.71531012e+00, 2.21326753e+00, 2.47608364e+00,
            2.46823614e+00, 2.22746396e+00, 1.85455234e+00, 1.48605305e+00,
            1.25732356e+00, 1.26588950e+00, 1.54504712e+00, 2.05485091e+00,
            2.69292577e+00, 3.32217715e+00, 3.80789874e+00, 4.05423385e+00,
            4.03012112e+00, 3.77770202e+00, 3.40091431e+00, 3.03735890e+00],
           [1.38121546e+00, 2.01514557e+00, 2.51310298e+00, 2.77591909e+00,
            2.76807159e+00, 2.52729941e+00, 2.15438779e+00, 1.78588850e+00,
            1.55715901e+00, 1.56572495e+00, 1.84488257e+00, 2.35468636e+00,
            2.99276122e+00, 3.62201260e+00, 4.10773419e+00, 4.35406930e+00,
            4.32995657e+00, 4.07753747e+00, 3.70074976e+00, 3.33719435e+00],
           [1.67255915e+00, 2.30648925e+00, 2.80444666e+00, 3.06726277e+00,
            3.05941528e+00, 2.81864309e+00, 2.44573147e+00, 2.07723218e+00,
            1.84850270e+00, 1.85706864e+00, 2.13622625e+00, 2.64603004e+00,
            3.28410491e+00, 3.91335628e+00, 4.39907787e+00, 4.64541298e+00,
            4.62130025e+00, 4.36888116e+00, 3.99209344e+00, 3.62853804e+00],
           [1.96847460e+00, 2.60240470e+00, 3.10036211e+00, 3.36317822e+00,
            3.35533073e+00, 3.11455854e+00, 2.74164692e+00, 2.37314763e+00,
            2.14441815e+00, 2.15298409e+00, 2.43214170e+00, 2.94194549e+00,
            3.58002036e+00, 4.20927173e+00, 4.69499332e+00, 4.94132843e+00,
            4.91721570e+00, 4.66479661e+00, 4.28800890e+00, 3.92445349e+00],
           [2.27566159e+00, 2.90959169e+00, 3.40754910e+00, 3.67036521e+00,
            3.66251772e+00, 3.42174554e+00, 3.04883391e+00, 2.68033462e+00,
            2.45160514e+00, 2.46017108e+00, 2.73932870e+00, 3.24913249e+00,
            3.88720735e+00, 4.51645872e+00, 5.00218032e+00, 5.24851542e+00,
            5.22440269e+00, 4.97198360e+00, 4.59519589e+00, 4.23164048e+00],
           [2.59810010e+00, 3.23203020e+00, 3.72998761e+00, 3.99280372e+00,
            3.98495623e+00, 3.74418405e+00, 3.37127242e+00, 3.00277313e+00,
            2.77404365e+00, 2.78260959e+00, 3.06176721e+00, 3.57157100e+00,
            4.20964586e+00, 4.83889723e+00, 5.32461883e+00, 5.57095393e+00,
            5.54684120e+00, 5.29442211e+00, 4.91763440e+00, 4.55407899e+00],
           [2.93838330e+00, 3.57231340e+00, 4.07027081e+00, 4.33308692e+00,
            4.32523942e+00, 4.08446724e+00, 3.71155562e+00, 3.34305633e+00,
            3.11432684e+00, 3.12289279e+00, 3.40205040e+00, 3.91185419e+00,
            4.54992906e+00, 5.17918043e+00, 5.66490202e+00, 5.91123713e+00,
            5.88712440e+00, 5.63470531e+00, 5.25791759e+00, 4.89436219e+00],
           [3.29831241e+00, 3.93224251e+00, 4.43019992e+00, 4.69301603e+00,
            4.68516853e+00, 4.44439635e+00, 4.07148473e+00, 3.70298544e+00,
            3.47425595e+00, 3.48282190e+00, 3.76197951e+00, 4.27178330e+00,
            4.90985817e+00, 5.53910954e+00, 6.02483113e+00, 6.27116624e+00,
            6.24705351e+00, 5.99463442e+00, 5.61784670e+00, 5.25429130e+00]])




```python
fig=plt.figure(figsize=(9,6))
ax=fig.gca(projection='3d')
surf1=ax.plot_surface(X,Y,Z,rstride=2,cstride=2,
                     cmap=mpl.cm.coolwarm,linewidth=0.5,
                     antialiased=True)
surf2=ax.plot_wireframe(X,Y,RZ,rstride=2,cstride=2,
                       label='regression')
ax.set_xlabel='x'
ax.set_ylabel='y'
ax.set_zlabel='f(x,y)'
ax.legend()
fig.colorbar(surf,shrink=0.5,aspect=5)
#align_figures()
```




    <matplotlib.colorbar.Colorbar at 0x1c2389d748>




![png](output_46_1.png)


# 2.插值


```python
import scipy.interpolate as spi
```


```python
x=np.linspace(-2*np.pi,2*np.pi,25)
x
```




    array([-6.28318531, -5.75958653, -5.23598776, -4.71238898, -4.1887902 ,
           -3.66519143, -3.14159265, -2.61799388, -2.0943951 , -1.57079633,
           -1.04719755, -0.52359878,  0.        ,  0.52359878,  1.04719755,
            1.57079633,  2.0943951 ,  2.61799388,  3.14159265,  3.66519143,
            4.1887902 ,  4.71238898,  5.23598776,  5.75958653,  6.28318531])




```python
def f(x):
    return np.sin(x)+0.5*x
f(x)
```




    array([-3.14159265, -2.37979327, -1.75196847, -1.35619449, -1.2283697 ,
           -1.33259571, -1.57079633, -1.80899694, -1.91322295, -1.78539816,
           -1.38962418, -0.76179939,  0.        ,  0.76179939,  1.38962418,
            1.78539816,  1.91322295,  1.80899694,  1.57079633,  1.33259571,
            1.2283697 ,  1.35619449,  1.75196847,  2.37979327,  3.14159265])




```python
ipo=spi.splrep(x,f(x),k=1) #插值
ipo
```




    (array([-6.28318531, -6.28318531, -5.75958653, -5.23598776, -4.71238898,
            -4.1887902 , -3.66519143, -3.14159265, -2.61799388, -2.0943951 ,
            -1.57079633, -1.04719755, -0.52359878,  0.        ,  0.52359878,
             1.04719755,  1.57079633,  2.0943951 ,  2.61799388,  3.14159265,
             3.66519143,  4.1887902 ,  4.71238898,  5.23598776,  5.75958653,
             6.28318531,  6.28318531]),
     array([-3.14159265, -2.37979327, -1.75196847, -1.35619449, -1.2283697 ,
            -1.33259571, -1.57079633, -1.80899694, -1.91322295, -1.78539816,
            -1.38962418, -0.76179939,  0.        ,  0.76179939,  1.38962418,
             1.78539816,  1.91322295,  1.80899694,  1.57079633,  1.33259571,
             1.2283697 ,  1.35619449,  1.75196847,  2.37979327,  3.14159265,
             0.        ,  0.        ]),
     1)




```python
iy=spi.splev(x,ipo) #Evaluate a B-spline or its derivatives.展现二维曲线插值
iy
```




    array([-3.14159265, -2.37979327, -1.75196847, -1.35619449, -1.2283697 ,
           -1.33259571, -1.57079633, -1.80899694, -1.91322295, -1.78539816,
           -1.38962418, -0.76179939,  0.        ,  0.76179939,  1.38962418,
            1.78539816,  1.91322295,  1.80899694,  1.57079633,  1.33259571,
            1.2283697 ,  1.35619449,  1.75196847,  2.37979327,  3.14159265])




```python
plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, iy, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
#align_figures()
```


![png](output_53_0.png)



```python
np.allclose(f(x),iy)
```




    True




```python
xd=np.linspace(1.0,3.0,50)
iyd=spi.splev(xd,ipo)
iyd
```




    array([1.33303162, 1.38197273, 1.41565273, 1.44650467, 1.47735662,
           1.50820856, 1.53906051, 1.56991245, 1.6007644 , 1.63161634,
           1.66246828, 1.69332023, 1.72417217, 1.75502412, 1.78555251,
           1.79551689, 1.80548128, 1.81544566, 1.82541004, 1.83537442,
           1.84533881, 1.85530319, 1.86526757, 1.87523195, 1.88519634,
           1.89516072, 1.9051251 , 1.91170102, 1.90357625, 1.89545147,
           1.88732669, 1.87920192, 1.87107714, 1.86295236, 1.85482759,
           1.84670281, 1.83857803, 1.83045326, 1.82232848, 1.8142037 ,
           1.80232804, 1.78375948, 1.76519093, 1.74662237, 1.72805381,
           1.70948525, 1.6909167 , 1.67234814, 1.65377958, 1.63521102])




```python
plt.plot(xd, f(xd), 'b', label='f(x)')
plt.plot(xd, iyd, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
```


![png](output_56_0.png)



```python
ipo=spi.splrep(x,f(x),k=3)
iyd=spi.splev(xd,ipo)
```


```python
plt.plot(xd, f(xd), 'b', label='f(x)')
plt.plot(xd, iyd, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid=True
plt.xlabel='x'
plt.ylabel='f(x)'
```


![png](output_58_0.png)



```python
np.allclose(f(xd),iyd)
```




    False




```python
np.sum((f(xd)-iyd)**2)/len(xd)
```




    1.1349319851436892e-08



# 3.Convex Optimization


```python
def fm(xy):
    x,y=xy
    return (np.sin(x)+0.05*x**2
           +np.sin(y)+0.05*y**2)
```


```python
x=np.linspace(-10,10,50)
y=np.linspace(-10,10,50)
X,Y=np.meshgrid(x,y)
XY=X,Y
Z=fm(XY)
```


```python
fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
cmap=mpl.cm.coolwarm,
linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
#align_figures()
```




    <matplotlib.colorbar.Colorbar at 0x1c2293f898>




![png](output_64_1.png)


## 3.1全局最优 Global Optimization


```python
import scipy.optimize as spo
```


```python
def fo(xy):  #"tuple parameter unpacking", was removed in Python 3.
    x,y=xy
    z=np.sin(x)+0.05*x**2+np.sin(y)+0.05*y**2
    if output==True:
        print('%8.4f %8.4f %8.4f'%(x,y,z))
    return z
```


```python
output=True
spo.brute(fo,((-10,10.1,5),(-10,10.1,5)),finish=None)
```

    -10.0000 -10.0000  11.0880
    -10.0000  -5.0000   7.7529
    -10.0000   0.0000   5.5440
    -10.0000   5.0000   5.8351
    -10.0000  10.0000  10.0000
     -5.0000 -10.0000   7.7529
     -5.0000  -5.0000   4.4178
     -5.0000   0.0000   2.2089
     -5.0000   5.0000   2.5000
     -5.0000  10.0000   6.6649
      0.0000 -10.0000   5.5440
      0.0000  -5.0000   2.2089
      0.0000   0.0000   0.0000
      0.0000   5.0000   0.2911
      0.0000  10.0000   4.4560
      5.0000 -10.0000   5.8351
      5.0000  -5.0000   2.5000
      5.0000   0.0000   0.2911
      5.0000   5.0000   0.5822
      5.0000  10.0000   4.7471
     10.0000 -10.0000  10.0000
     10.0000  -5.0000   6.6649
     10.0000   0.0000   4.4560
     10.0000   5.0000   4.7471
     10.0000  10.0000   8.9120





    array([0., 0.])




```python
output=False
opt1 = spo.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), finish=None)
opt1
```




    array([-1.4, -1.4])




```python
fm(opt1)
```




    -1.7748994599769203




```python
fo(opt1)
```




    -1.7748994599769203



## 3.2 局部最优 Local Optimization


```python
output=True
opt2=spo.fmin(fo,opt1,xtol=0.001,ftol=0.001,maxiter=15,maxfun=20)
opt2
```

     -1.4000  -1.4000  -1.7749
     -1.4700  -1.4000  -1.7743
     -1.4000  -1.4700  -1.7743
     -1.3300  -1.4700  -1.7696
     -1.4350  -1.4175  -1.7756
     -1.4350  -1.3475  -1.7722
     -1.4088  -1.4394  -1.7755
     -1.4438  -1.4569  -1.7751
     -1.4328  -1.4427  -1.7756
     -1.4591  -1.4208  -1.7752
     -1.4213  -1.4347  -1.7757
     -1.4235  -1.4096  -1.7755
     -1.4305  -1.4344  -1.7757
     -1.4168  -1.4516  -1.7753
     -1.4305  -1.4260  -1.7757
     -1.4396  -1.4257  -1.7756
     -1.4259  -1.4325  -1.7757
     -1.4259  -1.4241  -1.7757
     -1.4304  -1.4177  -1.7757
     -1.4270  -1.4288  -1.7757
    Warning: Maximum number of function evaluations has been exceeded.





    array([-1.42702972, -1.42876755])




```python
fm(opt2)
```




    -1.7757246992239009




```python
output=False
spo.fmin(fo,(2.0,2.0),maxiter=250)
```

    Optimization terminated successfully.
             Current function value: 0.015826
             Iterations: 46
             Function evaluations: 86





    array([4.2710728 , 4.27106945])



## 3.3限制最优 Constrained Optimization
#### 如，在预算限制下，求效用函数最大值（负效用函数最小值）


```python
#function to be minimized
from math import sqrt
def Eu(sb):
    s,b=sb
    return -(0.5*sqrt(s*15+b*5)+0.5*sqrt(s*5+b*12))

#constraints
cons=({'type':'ineq','fun':lambda sb: 100-10*sb[0]-10*sb[1]})
 #budget constraint
bnds=((0,1000),(0,1000)) #upper bounds large enough
```


```python
result=spo.minimize(Eu,[5,5],method='SLSQP',     #add an initial guess for the optimal parameters
                   bounds=bnds,constraints=cons)
result
```




         fun: -9.700883611487832
         jac: array([-0.48508096, -0.48489535])
     message: 'Optimization terminated successfully.'
        nfev: 21
         nit: 5
        njev: 5
      status: 0
     success: True
           x: array([8.02547122, 1.97452878])




```python
#a和b的最优解
result['x']
```




    array([8.02547122, 1.97452878])




```python
#效用最大值
-result['fun']
```




    9.700883611487832




```python
#初始成本
np.dot(result['x'],[10,10])
```




    99.99999999999999



# 4.积分 Integration


```python
import scipy.integrate as sci
```


```python
def f(x):
    return np.sin(x)+0.5*x
```


```python
a=0.5 #left integral limit
b=9.5 #right integral limit
x=np.linspace(0,10)
y=f(x)
```


```python
from matplotlib.patches import Polygon
fig,ax=plt.subplots(figsize=(7,5))
plt.plot(x,y,'b',linewidth=2)
plt.ylim(ymin=0)

#area under the function
#between lower and upper limit
Ix=np.linspace(a,b)
Iy=f(Ix)
#verts=[(a,0)+list(zip(Ix,Iy))+[(b,0)]]
verts=[(a,0)]+list(zip(Ix,Iy))+[(b,0)]
poly=Polygon(verts,facecolor='0.7',edgecolor='0.5')
ax.add_patch(poly)


#labels
plt.text(0.75*(a+b),1.5,r"$\int_a^b f(x)dx$",
        horizontalalignment='center',fontsize=20) #积分公式

plt.figtext(0.9,0.075,'$x$') #横轴角标
plt.figtext(0.075,0.9,'$f(x)$') #纵轴角标

ax.set_xticks((a,b)) #指定x坐标轴刻度数值为a,b
ax.set_xticklabels(('$a$','$b$')) #指定x坐标轴刻度不显示为数值，而显示为a,b
ax.set_yticks([f(a),f(b)]) #指定y坐标轴刻度显示为数值

#align_figures()

```




    [<matplotlib.axis.YTick at 0x1c20ff1c88>,
     <matplotlib.axis.YTick at 0x1c20ff1438>]




![png](output_86_1.png)



```python
#求积分
sci.fixed_quad(f, a, b)[0]
```




    24.366995967084602




```python
for i in range(1,20):
    np.random.seed(1000)
    x=np.random.random(i*10)*(b-a)+a #x为数值介于[a,b]的随机list
    print(np.sum(f(x))/len(x)*(b-a)) #每个x的list的平均f(x)为高，[a,b]为长，求出的矩形面积
```

    24.804762279331463
    26.522918898332378
    26.265547519223976
    26.02770339943824
    24.99954181440844
    23.881810141621663
    23.527912274843253
    23.507857658961207
    23.67236746066989
    23.679410416062886
    24.424401707879305
    24.239005346819056
    24.115396924962802
    24.424191987566726
    23.924933080533783
    24.19484212027875
    24.117348378249833
    24.100690929662274
    23.76905109847816


# 5.代数式Symbolic Computation


```python
import sympy as sy
```


```python
x=sy.Symbol('x')
y=sy.Symbol('y')
```


```python
#look up birthday in pi
pi_str=str(sy.N(sy.pi, 400000))
pi_str.find('0927')
```




    1177




```python
pi_str[1177:1181]
```




    '0927'



## 5.1解方程


```python
sy.solve(x**2-1)
```




    [-1, 1]



## 5.2积分


```python
a,b=sy.symbols('a b')
```


```python
print(sy.pretty(sy.Integral(sy.sin(x)+0.5*x,(x,a,b))))
```

    b                    
    ⌠                    
    ⎮ (0.5⋅x + sin(x)) dx
    ⌡                    
    a                    


### 方法一


```python
int_func=sy.integrate(sy.sin(x)+0.5*x,x)
print(sy.pretty(int_func))
```

          2         
    0.25⋅x  - cos(x)



```python
Fb=int_func.subs(x,9.5).evalf()
Fa=int_func.subs(x,0.5).evalf()
```


```python
Fb-Fa # exact value of integral
```




$\displaystyle 24.3747547180867$



### 方法二


```python
int_func_limits=sy.integrate(sy.sin(x)+0.5*x,(x,a,b))
print(sy.pretty(int_func_limits))
```

            2         2                  
    - 0.25⋅a  + 0.25⋅b  + cos(a) - cos(b)



```python
int_func_limits.subs({a:0.5,b:9.5}).evalf()
```




$\displaystyle 24.3747547180868$



### 方法三


```python
sy.integrate(sy.sin(x)+0.5*x,(x,0.5,9.5))
```




$\displaystyle 24.3747547180867$



## 5.3微分


```python
int_func.diff()
```




$\displaystyle 0.5 x + \sin{\left(x \right)}$



### 求极值


```python
f=(sy.sin(x)+0.05*x**2
  +sy.sin(y)+0.05*y**2)
```


```python
#偏导
del_x=sy.diff(f,x)
del_y=sy.diff(f,y)
print(sy.pretty(del_y))
```

    0.1⋅y + cos(y)



```python
#最值的educated guess
xo=sy.nsolve(del_x,-1.5)
yo=sy.nsolve(del_y,-1.5)
print(sy.pretty(yo))
```

    -1.42755177876459



```python
min=f.subs({x:xo,y:yo}).evalf() #global minimum
print(sy.pretty(min))
```

    -1.77572565314742

