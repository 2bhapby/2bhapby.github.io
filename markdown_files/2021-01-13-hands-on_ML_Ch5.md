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
-- 


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk5OTQyODI2MywxMzA2ODc2NTIxLC0xOD
kzMDIxMzE5LC0xMjkzODk2NDIyLDE3OTM2OTgyOTNdfQ==
-->