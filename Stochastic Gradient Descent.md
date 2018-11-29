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



