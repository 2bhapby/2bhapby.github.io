---
layout: post
title: 모델훈련
---


# 서포트 벡터 머신

- 복잡한 분류 문제, 작거나 중간 크기의 데이터셋에 적합하다
- 특성의 스케일에 민감하다

## 선형 SVM 분류

**SVM 분류기**
- 클래스 사이에 가장 폭이 넓은 도로를 찾는 것 **(라지 마진 분류)**


**서포트 벡터**
- 도로 바깥쪽에 훈련 샘플을 더 추가해도 결정 경계에는 전혀 영향을 미치지 않는다.
- 도로 경계에 위치한 샘플에 의해 전적으로 결정된다.

위와 같은 샘플을 서포트 벡터라고 한다.


### 소프트 마진 분류


**하드 마진 분류(hard margin classification)**

- 모든 샘플이 도로 바깥쪽에 올바르게 분류
- 문제점
--데이터가 선형적으로 구분될 수 있어야 함
--이상치에 민감

**소프트 마진 분류(soft margin classification)**

- 도로의 폭을 가능한 넓게 유지하는 것과 마진 오류사이에 적절한 균형을 잡는 것

**사이킷런의 SVM 모델**
- 하이퍼 파라미터 : C
-- 마진 오류의 정도를 잡아주는 것(숫자가 작으면 넓은 마진, 숫자가 크면 좁은 마진)
-- 과대적합이면 C를 감소시켜 모델을 규제
-- 클래스에 대한 확률을 제공하지 않는다

- LinearSVC 클래스를 대신 선형 커널을 사용하는 SVC 클래스로 대체할 수 있다.
- SVC(kernel = 'linear', C = 1)이라고 쓴다.
- SGDClassifier(loss = 'hinge', alpha = 1/(m*C)) 로 표현한다.
- 선형 SVM 분류기를 훈련시키기 위해서 일반적인 확률적 경사 하강법을 적용한다.
-- LinearSVC 만큼 빠르게 수렴하진 않지만 데이터셋이 아주 커서 메모리에 적재할 수 없거나 온라인 학습 분류 문제를 다룰 때 유용하다.
-- LinearSVC는 규제에 편향을 포함시킨다. 그래서 훈련 세트에서 평균을 빼서 중앙에 맞춰야한다. (StandardScaler 사용)
-- loss 매개변수는 hinge 지정
-- 훈련 샘플보다 특성이 많지 않다면 성능을 높이기 위해 dual 매개변수를 False로 지정.

## 비선형 SVM 분류

비선형 데이터를 다루는 방법
- 다항 특성과 같은 특성을 더 추가한다. (ex. x^2 항을 추가한다)

### 다항식 커널

높은 차수의 다항식은 느리고 낮은 차수의 다항식은 데이터 셋을 잘 표현하지 못한다.

SVM은 **커널 트릭**을 적용할 수 있다.
>커널 트릭은 실제로 특성을 추가하지 않으면서 다항식 특성을 많이 추가한 것과 같은 결과를 얻을 수 있다.


("svm_clf", SVC(kernel = 'poly', degree = 3, coef0 = 1, C = 5))
- coef0는 모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 조절한다.
- 차수가 높으면 과대적합되기 쉽고 차수가 낮으면 과소적합되기 싶다.

### 유사도 특성 

- 각 샘플이 특정 랜드마크와 얼마나 닮았는지 측정하는 유사도 함수로 계산한 특성을 추가하는 것
- 가우시안 방사 기저 함수를 유사도 함수로 정의
--$\phi_\gamma(\textbf x, l) = exp()$
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTczMjY3NDA3OCwxNTkxNzExODg1LDI0Mj
c1MDcsOTg2NjE4MTUwLDcwMDYwNjA2MSwtNjM0Nzg4NTc5LDg2
ODQ5NTIyNSwtNjk5MjM5NTY0LDE1Mjc2Mjk1OTcsMTMwNjg3Nj
UyMSwtMTg5MzAyMTMxOSwtMTI5Mzg5NjQyMiwxNzkzNjk4Mjkz
XX0=
-->