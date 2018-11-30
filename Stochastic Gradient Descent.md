# 1.5. Stochastic Gradient Descent (확률적 경사 하강)

확률적 경사 하강(SGD)은 지원 벡터 머신(SVM) 및 로지스틱 회귀 분석(LM) 같은 블록 손실 함수에서 선형 분류기의 차별적 학습에 간단하지만 매우 효율적인 방법이다.

SGD는 텍스트 분류와 자연 언어 처리에서 종종 발생하는 대용량 및 희소 기계 학습 문제에 성공적으로 적용되었다. 데이터가 희박하다면 이 모듈의 분류자는 10^5 이상의 학습 예제와 10^5 이상의 기능을 가진 문제까지 쉽게 확장된다.

SGD의 장점은 다음과 같다.

- 효율성
- 구현의 용이성

단점은 다음과 같다.

- SGD는 정규화 매개 변수와 반복 횟수 같은 많은 매개 변수가 필요하다.
- 기능 확장에 민감하다.

## 1.5.1. Classification (분류)

`SGDClassifier` 클래스는 다양한 손실 기능과 분류에 대한 벌칙을 지원하는 단순한 확률적 경사 하강 학습 루틴을 구현한다.

SGD에는 학습 샘플을 보관하는 크기 [n_samples, n_Features]의 배열 X와 훈련 샘플의 대상 값을 포함하는 크기 [n_samples]의 배열 Y 이렇게 두 가지 배열이 있어야 한다.

```python
>>> from sklearn.linear_model import SGDClassifier
>>> X = [[0., 0.], [1., 1.]]
>>> y = [0, 1]
>>> clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
>>> clf.fit(X, y)   
SGDClassifier(alpha=0.0001, average=False, class_weight=None,
           early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
           l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=5,
           n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',
           power_t=0.5, random_state=None, shuffle=True, tol=None,
           validation_fraction=0.1, verbose=0, warm_start=False)
```

그런 다음 모델을 사용하여 새 값을 예측할 수 있다.

```python
>>> clf.predict([[2., 2.]])
array([1])
```

SGD는 훈련 데이터에 선형 모델을 적용한다. 멤버 `coef_`는 모델 매개 변수를 보유한다.

```python
>>> clf.coef_                                         
array([[9.9..., 9.9...]])
```

멤버 `intercept_`는 오프셋을 보유한다.

```python
>>> clf.intercept_                                    
array([-9.9...])
```

모델이 오프셋을 사용해야 하는지 여부는 `fit_intercept`에 의해 제어된다.

```python
>>> clf.decision_function([[2., 2.]])                 
array([29.6...])
```

---

## 1.5.2. Regression (회귀 분석)

`SGDRegressor` 클래스는 다양한 손실 기능과 벌칙을 지원하는 단순 확률적 경사 하강 학습 루틴을 구현한다. `SGDRegressor`는 많은 훈련 샘플(10.000 이상)의 회귀 문제에 적합하며, 다른 문제 경우에는 `Ridge`, `Rasso`또는 `ElasticNet`을 권장한다.

콘크리트 손실 기능은 손실 매개변수를 통해 설정할 수 있다. `SGDRegressor`는 다음과 같은 손실 기능을 지원한다.

- `loss="squared_loss"`: 보통 최소 제곱
- `loss="huber"`: 강력한 회귀 분석을 위한 휴버 손실
- `loss="epsilon_insensitive"`: 선형 지원 벡터 회귀 분석

강령한 회귀 분석에 휴버 및 엡실론 손실 기능을 사용할 수 있다. 민감하지 않은 영역의 너비는 매개 변수 엡실론을 통해 지정해야 한다. 이 매개 변수는 대상 변수의 축적에 따라 달라진다.

`SGDRegressor`는 `SGDClassifier`로 평균 SGD를 지원합니다. `average=True`을 설정하여 평균값을 설정할 수 있다.

---

## 1.5.3. Stochastic Gradient Descent for sparse data (스파스 데이터에 대한 확률적 그라데이션 강하)

Note: 스파스 구현은 인터셉트의 훈력 속도가 죽어들기 때문에 조밀한 구현과 약간 다른 결과를 산출한다.

scipy.sparse가 지원하는 형식으로 모든 행렬에 희소 데이터에 대한 기본 지원이 있다. 그러나 효율성을 극대화 하려면 `scipy.sparse.csr_matrix`에 정의 된대로 CSR 행렬 형식을 사용해야한다.

---

## 1.5.4. Complexity (복잡성)

SGD의 가장 큰 장점은 효율성인데, 이는 기본적으로 훈련의 수에 있어 선형적이다. 만약 X가 크기 (n, p)의 행렬이라면 훈련 비용은 O(knp)다. 여기서 k는 반복 횟수이며 샘플당 평균 0이 아닌 속성 수다.

그러나 최근 이론적인 결과는 훈련 세트 크기가 증가함에 따라 원하는 최적의 정확도를 얻기 위한 런타임이 증가하지 않는다는 것을 보여준다.

---

## 1.5.5. Stopping criterion (중지 기준)

`SGDClassifier`와 `SGDRegressor` 클래스는 주어진 융합 수준에 도달할 때 알고리즘을 정지하기 위한 두 가지 기준을 제공한다.

- `early_stopping=True`를 사용하면 입력 데이터가 훈련 세트와 검증 세트로 분할된다. 그런 다음 모델은 훈련 세트에 장착되며, 정지 기준은 검증 세트에서 계산한 예측 점수를 기반으로 한다. 검증 세트의 크기는 파라미터 `verification_fraction`을 사용하여 변경할 수 있다.
- `early_stopping=False`인 경우 모델은 전체 입력 데이터에 적합하며, 정지 기준은 입력 데이터를 기반으로 계산된 목표 함수에 기초한다.

위 두가지 경우 모두 기준은 한 시대마다(once by epoch) 한 번 평가되며, 기준이 한 행에서 `n_iter_no_change` 시간을 개선하지 않으면 알고리즘이 중단된다. 개선 사항은 오차 `tol`에 의해 평가되며, 알고리즘은 최대 반복 `max_itter` 수 이후 무조건 정지한다.

---

## 1.5.6. Tips on Practical Use (실제 사용에 대한 팁)

확률적 경사 하강은 기능 확장에 민감으로 데이터를 확장하는 것이 좋다. 예를 들어, 입력 베터 X의 각 특서을 [0, 1]  또는 [-1, +1]로 스케일링 하거나 평균 0과 분산 1로 표준화 한다. 의미 있는 결과를 얻으려면 텍스트 벡터에 동일한 스케일링을 적용해야 한다. 이 작업은 `StandardScaler`를 사용해서 쉽게 수행할 수 있다.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Don't cheat - fit only on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # apply same transformation to test data
```

---

수학 공식과 구현 세부 사항은 아래 링크에서 확인할 수 있다.

https://scikit-learn.org/stable/modules/sgd.html#mathematical-formulation

https://scikit-learn.org/stable/modules/sgd.html#implementation-details

---

# SVM과 기능적 차이

지원 벡터 머신(SVM)은 고차원 공간에서 효과적이며, 지원 벡터에서 학습 지점의 일부를 사용하므로 효율이 높다. Sklearn의 SVM은 librinal 및 libsvm을 기반으로한다.

확률적 경사 하강(SGD)은 SVM만큼 빠르지 않지만 데이터가 아주 많아서 메모리에 적재할 수 없을때 유용하다. 대용량 데이터에 대해 더 잘 확장 되고, 많은 매개 변수가 필요하다. 또한 커널을 지정할 필요가 없다.