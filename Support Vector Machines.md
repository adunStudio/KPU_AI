# 1.4. Support Vector Machines (지원 벡터 머신)

**지원 벡터 머신(SVMs)은 분류, 회귀 및 특이치 검출을 위해 사용되는 감독된 학습 방법 집합이다.** 
지원 벡터 머신의 장점은 다음과 같다.

- 고차원 공간에서 효과적이다.
- 차원 수가 표본 수보다 큰 경우에도 효과적이다.
- 지원 벡터라고 불리는 결정 함수에서 학습 지점의 일부를 사용하므로 메모리 효율도 높다.
- 결정 함수를 위해 다른 커널 함수를 지정할 수 있다. 공용 커널이 제공되지만 사용자 지정 커널을 지정할 수도 있다.

지원 벡터 머신의 단점은 다음과 같다.

- 형상의 수가 표본의 수보다 훨씬 많으면 무리가 따르지 않도록 커널 함수와 정규화 기간을 선택하는게 중요하다.
- SVM은 확률 산정을 직접 제공하지 않으며, 이는 값비싼 5배 교차 검증을 사용해서 계산된다.(아래의 점수 및 확률 참조)

scikit-learn의 SVM은 입력으로 밀도가 높은 표본 벡터(`numfy.ndarray`와 변환 가능한 `numpy.asarray`)와 희소 표본 벡터(모든 `scipy.sparse`)를 모두 지원한다. 그러나 SVM을 사용해서 희소 데이터를 예측하려면 적합한 데이터를 사용해야 한다. 성능 최적화를 위해서는 C-orcered  `numpy.ndarray`(밀집) 또는  `scipy.spare.csr_matrix` (희소)와 함께 `dtype-float64`를 사용해야 한다.

---

## 1.4.1. Classification (분류)

`SVC`, `NuSVC` 및 `LinearSVC`는 데이터셋에서 다중 클래스 분류를 수행할 수 있는 클래스다.

![01](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_0012.png)

`SVC`와 `NuSVC`는 비슷한 메서드이지만 약간 다른 매개 변수 세트를 허용하고 다른 수학 공식을 가진다(수학 공식 세션 참조). 반면 `LinearSVC`는 선형 커널의 경우에 대한 SVC의 또 다른 구현이다. `LinearSVC`는 선형이라는 가정하에 `kernel` 키워드를 허용하지 않는다. 또한 `support_`와 같은 `SVC`와 `NuSVC`의 일부 멤버도 부족하다.

다른 분류자인 `SVC`, `NuSBC` 및 `LinearSVC`는 두 가지 배열을 입력으로 사용한다.

- X: 학습 배열을 보유하는 `[n_samples, n_features]`의 크기, 클래스 레이블(문자열 또는 정수)

- y: 크기 `[n_samples]`

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

SVM의 결정 함수는 지원 벡터라고 하는 일부 학습 데이터에 따라 달라진다. 이러한 지원 벡터의 일부 속성은 멤버 `support_vetctors_`, `support_` 및 `n_support`에서 찾을 수 있다.

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

### 1.4.1.1 Multi-class classification (다중 클래스 분류)

`SVC`와 `NuSVC`는 다중 클래스 분류를 위해 "one-against-one" (Knerr et al., 1990)을 구현했다. `n_class`가 클래스의 수인 경우 `n_class * (n_class - 1) / 2`개의 분류자가 생성되고 각각은 두 클래스의 데이터를 훈련한다. 다른 분류자와 일관된 인터페이스를 제공하기 위해, `decision_function_shape` 옵션을 사용하면 "one-against-one" 분류 기준의 결과를 결정 함수의 형상`(n_samples, n_classes)`에 집계할 수 있다.

```python
>>> X = [[0], [1], [2], [3]]
>>> Y = [0, 1, 2, 3]
>>> clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
>>> clf.fit(X, Y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
>>> dec = clf.decision_function([[1]])
>>> dec.shape[1] # 4 classes: 4*3/2 = 6
6
>>> clf.decision_function_shape = "ovr"
>>> dec = clf.decision_function([[1]])
>>> dec.shape[1] # 4 classes
4
```

반면, `LinearSVC`는 "one-vs-rest" 다중 클래스 전략을 구현하여 `n_class` 모델을 훈련한다. 클래스가 두 개인 경우 하나의 모델만 훈련된다.

```cpp
>>> lin_clf = svm.LinearSVC()
>>> lin_clf.fit(X, Y) 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
>>> dec = lin_clf.decision_function([[1]])
>>> dec.shape[1]
4
```

결정 함수에 대한 자세한 설명은 1.4.7. 수학 공식을 참조하면 된다.

`LinearSVC`는 `multi_class='crammer_singer'` 옵션을 사용하여 Crammer와 Singer에 의해 구성된 다중 클래스 SVM이라고 불리는 대체 전략을 구현했다. 이 전략은 일관성이 있으며, "one-vs-rest" 분류에는 해당되지 않는다. 실제로 결과는 대부분 비슷하지만 "one-vs-rest" 분류가 런타임이 훨씬 적기 때문에  일반적으로 더 선호된다.

"one-vs-rest" LinearSVC의 경우 `coef_`와 `intercept_` 속성은 각각 `[n_class, n_features]`와 `[n_class]` 형상을 갖는다. 계수의 각 행은 `n_class` 많은 "one-vs-rest" 분류자 중 하나에 해당하며, 1등급 순서로 인터셉트와 유사하다.

"one-vs-one" SVC의 경우, 속성의 레이아웃이 조금 더 연관되어 있다. 선형 커널을 사용하는 경우 `coef_`와 `intercept_` 속성은 각각 `[n_class * (n_class - 1) / 2, n_features] `와 `[n_class * (n_class - 1) / 2] ` 형상을 갖는다. 이는 위에 설명된 LinearSVC의 레이아웃과 유사하며, 이제 각 행은 바이너리 분류기에 해당한다. 0에서 n까지의 등급은“ 0 vs 1”, “0 vs 2” , … “0 vs n”, “1 vs 2”, “1 vs 3”, “1 vs n”, . . . “n-1 vs n”이다.

`dual_coef`의 형상은 `[n_class-1, n_SV]`이며 레이아웃을 파악하기가 다소 어렵다.  열은 `n_class * (n_class - 1) / 2` "one-vs-one" 분류자에 관련된 지원 벡터에 해당한다. 각 지원 벡터는 `n_class - 1` 분류자에서 사용된다. 각 행의 `n_class - 1` 항목은 이러한 분류자에 대한 이중 계수에 해당한다.

이는 다음 예를 통해 더욱 명확히 설명할 수 있다.

다음과 같이 지원 벡터가 있는 클래스 3개를 고려해보자.

- 3개의 지원 벡터(v00, v01, v02)를 갖는 클래스 0
- 2개의 지원 벡터(v10, v11)를 갖는 클래스 1
- 2개의 지원 벡터(v20, v21)를 갖는 클래스 2

각각의 지원 벡터 vij에서, 각 지원 벡터에 대해 두 개의 이중 계수가 있다 클래스 i와 k 사이의 분류자에서 지원 벡터 vij의 계수를 호출해보자. 그러면 `dual_coef_`는 다음과 같다.

| α00,1α0,10 | α00,2α0,20 | Coefficients for SVs of class 0 |
| ---------- | ---------- | ------------------------------- |
| α10,1α0,11 | α10,2α0,21 |                                 |
| α20,1α0,12 | α20,2α0,22 |                                 |
| α01,0α1,00 | α01,2α1,20 | Coefficients for SVs of class 1 |
| α11,0α1,01 | α11,2α1,21 |                                 |
| α02,0α2,00 | α02,1α2,10 | Coefficients for SVs of class 2 |
| α12,0α2,01 | α12,1      |                                 |

---

### 1.4.1.2. Scores and probabilities (점수와 확률)

`SVC`와 `NuSVC`의 `decision_function` 메서드는 각 샘플에 대해 클래스당 점수를 제공한다 (또는 바이너리의 경우 샘플당 하나의 점수). 생성자 옵션 `probability`를 `True`로 설정하면 클래스 멤버 자격 추정치가 활성화 된다.(`predict_proba`와 `predict_log_proba` 메서드로부터)  바이너리 경우에는 Platt scaling을 사용하여 확률이 조정된다: SVM 점수에 대한 로지스틱 회귀 분석. 훈련 데이터에 대한 추가 교차 검증에 적합하다. 다중 클래스 경우에는 Wu등에 따라 확장된다 (2004).

말할 필요도없이, Platt 스케일링과 관련된 교차 검증은 대형 데이터 세트에 대한 비용이 많이 드는 작업이다. 또한, 확률 추정치는 "argmax"가 확률의 argmax가 아닐 수도 있다는 점에서 점수와 일치하지 않을 수 있다. Platt의 방법은 이론적인 문제도 가지고 있는 것으로 알려져 있다. 신뢰 점수가 필요하지만 이 점수가 확률일 필요는 없는 경우, `probability=False`을 설정하고 `predict_proba` 대신 `decision_function`을 사용하는 것이 좋다.

---

### 1.4.1.3. Unbalanced problems (불균형 문제)

특정 클래스 또는 특정 개별 샘플 키워드를 더 중요하게하는 문제에서 `class_weight`와  `sample_weight` 키워드를 사용할 수 있다.

`SVC`(`NuSVS` 제외)는 `fit` 메서드에서 키워드 `class_weight`를 구현했다. `{class_label : value}` 형식으 딕셔너리다. 여기서 value는 `class_label`의 매개 변수 `C`를 `C * value` 값으로 설정하는 0보다 큰 부동 소수점 숫자다.

![02](https://scikit-learn.org/stable/_images/sphx_glr_plot_separating_hyperplane_unbalanced_0011.png)

`SVC`, `NuSVC` , `SVR`, `NuSVB` 및 `OneClassSVM`도 키워드 `sample_wheight`를 통해 적합한 방법으로 개별 샘플에 대한 가중치를 구현했다. `class_weight`와 유사하게, 이들은 i 번째 예제의 매개 변수 C를 `C * sample_weight[i]`로 설정한다.

---

## 1.4.2. Regression (회귀 분석)

회귀 문제를 해결하기 위해 SVC 메서드를 화장할 수 있다. 이 메서드를 지원 벡터 회귀 분석(Support Vector Regression)이라고 한다.

지원 벡터 분류(support vector classification)에 의해 생성되는 모델은 모델을 구축하기 위한 비용 함수가 마진을 초과하는 훈련 지점에 대해서는 상관하지 않기 때문에 훈련 데이터의 하위 집합에만 의존한다. 마찬가지로 지원 벡터 회귀 분석에 의해 생성된 모델은 모델 예측에 가까운 훈련 데이터를 무시하기 때문에 훈련 데이터의 하위 집합에만 의존한다.

지원 벡터 회귀 분석에는 세 가지 구현 `SVR`, `NuSVR` 및 `LinearSVR`이 있다. `LinearSVR`은 `SVR`보다 구현 속도가 빠르지만 선형 커널만 고려한다. 자세한 내용은 구현 세부 정보를 참조하면 된다.

클래스 분류와 마찬가지로 `fit` 메서드는 인수 벡터 `X`, `y`로 취해지며, `y`는 정수 값 대신 부동 소수점 값을 가져야 한다.

```python
>>> from sklearn import svm
>>> X = [[0, 0], [2, 2]]
>>> y = [0.5, 2.5]
>>> clf = svm.SVR()
>>> clf.fit(X, y) 
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)
>>> clf.predict([[1, 1]])
array([1.5])
```

---

## 1.4.3. Density estimation, novelty detection (밀도 추정, 신규성 검출)

`OneClassSVM` 클래스는 특이치 탐지에 사용되는 One-Class SVM을 구현한다.

`OneClassSVM`의 설명 및 사용에 대한 자세한 내용은 Novelly 및 Outlier Detection을 참조하면 된다.

---

## 1.4.4. Complexity (복잡성)

지원 벡터 머신은 강력한 도구이지만 훈련 벡터의 수에 따라 컴퓨팅 및 스토리지 요구 사항이 급격히 증가한다. SVM의 핵심은 훈련 데이터의 나머지 부분에서 지원 벡터를 분리하는 2차 프로그래밍 문제(QP)다.

또한 선형 케이스의 경우 `liblinear` 구현에 의해 `LinearSVC`에 사용된 알고리즘은 `libsvm` 기반 SVC 카운터보다 훨씬 효율적이며 수백만개의 샘플 또는 기능으로 확장할 수 있다.

---

## 1.4.6. Kernel functions (커널 함수)

초기화 시 키워드 커널에 따라 다른 커널이 지정된다.

```python
>>> linear_svc = svm.SVC(kernel='linear')
>>> linear_svc.kernel
'linear'
>>> rbf_svc = svm.SVC(kernel='rbf')
>>> rbf_svc.kernel
'rbf'
```

### 1.4.6.1. Custom Kernels

커널을 파이썬 함수로 제공하거나 그램 행렬을 미리 계산하여 자신만의 커널을 정의할 수 있다.

사용자 지정 커널을 사용하는 분류자는 다음을 제외하고 다른 분류자와 동일하게 작동한다.

- 필드 `support_vectors_`가 현재 비어 있으며, 지원 벡터의 인덱스만 `support_`에 저장된다.
- `fit()` 메서드의 첫 번째 인수의 참조가 나중에 참조할 수 있도록 저장된다. 해당 배열이 `fit()` 또는 `predict()` 메서드 사용 사이에 변경되면 예기치 않은 결과가 발생한다.

#### 1.4.6.1.1. Using Python functions as kernels

생성자의 키워드 커널에 함수를 전달하여 정의된 커널을 사용할 수도 있다.

```python
>>> import numpy as np
>>> from sklearn import svm
>>> def my_kernel(X, Y):
...     return np.dot(X, Y.T)
...
>>> clf = svm.SVC(kernel=my_kernel)
```

---

수학 공식과 구현 세부 사항은 아래 링크에서 확인할 수 있다.

https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation

https://scikit-learn.org/stable/modules/svm.html#implementation-details