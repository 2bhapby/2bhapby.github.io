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

- 도로의 폭을 가능한 넓게 유지하는 것과 마진 오류사이에 적절한 균형을 잡아야 한다.



<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc5MzY5ODI5MywxMjAyODU1NzY4LDE0Mj
Q4NzA1NywyMTMwNzgzMTc2XX0=
-->