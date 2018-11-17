# 1.4. Support Vector Machines (지원 벡터 머신)

**지원 벡터 머신(SVMs)은 부류, 회귀 및 특이치 검출을 위해 사용되는 감독된 학습 방법 집합이다.** 
지원 벡터 머신의 장점은 다음과 같다.

- 고차원 공간에서 효과적이다.
- 차원 수가 표본 수보다 큰 경우에도 효과적이다.
- 지원 벡터라고 불리는 의사 결정 기능에서 학습 지점의 일부를 사용하므로 메모리 효율도 높다.
- 의사 결정 기능을 위해 다른 커널 기능을 지정할 수 있다. 공용 커널이 제공되지만 사용자 지정 커널을 지정할 수도 있다.

지원 벡터 머신의 단점은 다음과 같다.

- 형상의 수가 표본의 수보다 훨씬 많으면 무리가 따르지 않도록 커널 기능과 정규화 기간을 선택하는게 중요하다.
- SVM은 확률 산정을 직접 제공하지 않으며, 이는 값비싼 5배 교차 검증을 사용해서 계산된다.(아래의 점수 및 확률 참조)

scikit-learn의 SVM은 입력으로 밀도가 높은 표본 벡터(`numfy.ndarray`와 변환 가능한 `numpy.asarray`)와 희소 표본 벡터(모든 `scipy.sparse`)를 모두 지원한다. 그러나 SVM을 사용해서 희소 데이터를 예측하려면 적합한 데이터를 사용해야 한다. 성능 최적화를 위해서는 C-orcered  `numpy.ndarray`(밀집) 또는  `scipy.spare.csr_matrix` (희소)와 함께 `dtype-float64`를 사용해야 한다.

---

## 1.4.1. Classification (분류자)

`SVC`, `NuSVC` 및 `LinearSVC`는 데이터셋에서 다중 클래스 분류를 수행할 수 있는 클래스다.

![01](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_0012.png)

`SVC`와 `NuSVC`는 비슷한 메서드이지만 약간 다른 매개 변수 세트를 허용하고 다른 수학 공식을 가진다(수학 공식 세션 참조). 반면 `LinearSVC`는 선형 커널의 경우에 대한 SVC의 또 다른 구현이다. `LinearSVC`는 선형이라는 가정하에 `kernel` 키워드를 허용하지 않는다. 또한 `support_`와 같은 `SVC`와 `NuSVC`의 일부 멤버도 부족하다.

다른 분류자인 `SVC`, `NuSBC` 및 `LinearSVC`는 두 가지 배열을 입력으로 사용한다.

- 학습 배열을 보유하는 `[n_samples, n_features]`의 크기, 클래스 레이블(문자열 또는 정수)

- 크기: `[n_samples]`

```python
>>> from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC(gamma='scale')
>>> clf.fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

그런 다음 모델을 사용하여 새 값을 예측하는 데 사용될 수 있다.

```python
>>> clf.predict([[2., 2.]])
array([1])
```

SVM의 의사 결정 기능은 지원 벡터라고 하는 일부 학습 데이터에 따라 달라진다. 이러한 지원 벡터의 일부 속성은 멤버 `support_vetctors_`, `support_` 및 `n_support`에서 찾을 수 있다.

```python
>>> # get support vectors
>>> clf.support_vectors_
array([[0., 0.],
       [1., 1.]])
>>> # get indices of support vectors
>>> clf.support_ 
array([0, 1]...)
>>> # get number of support vectors for each class
>>> clf.n_support_ 
array([1, 1]...)
```

---

