# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 박영준
- 리뷰어 : 조대희


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 특정 기능을 하는 각각의 코드에 주석을 달아 이해하기 용이했습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 아래 작성자 코드에 기재함.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 해당 내용을 충분히 인지하고 설명하였습니다.
- [X] 코드가 간결한가요?
  > 크게 개선될 부분없이 가결해 보입니다. 다만 모든 주석이 각 코드의 위에 배치되어있어, 가독성이 떨어질 우려가 있어보입니다.
  > 때에따라 인라인 주석(해당코드 옆 주석)을 사용하여 보다 가독성을 높힐 수 있을 것 같습니다.

# 예시

### ChatGPT 프롬프트 명령어

```python
# Q4. 물고기 정보(작성자 코드)
import time as t

fish_list = [
{"이름": "Nemo", "speed": 3},
{"이름": "Dory", "speed": 5},
]

# 컴프리헨션 함수 지정
def show_fish_movement_comprehension(fish_list):

    # 출력값 변수에 fish_list를 반복 출력하여 '이름'과 'speed' 한 쌍으로 묶이는 리스트를 만들기
    출력값 = [(fish['이름'],fish['speed']) for fish in fish_list]

    # 반복문으로 출력값에 들어있는 [name, speed]를 각각 변수로 만들기
    for name,speed in 출력값:

      # print 출력 - 반복문으로 만든 name 과 speed를 문자열과 같이 출력
      print(name,'is swimming at',speed, 'm/s')

      # time 모듈에 sleep을 실행하여 2초 간격두기
      t.sleep(2)

# 제너레이터 함수 지정
def show_fish_movement_Generator(fish_list):

    # 반복문으로 fish_list 딕셔너리 형태로 fish 변수에 할당
    for fish in fish_list:

        # f{} 포맷 형태로 yield 지정하여 func에 할당시켜두기
        yield f'''{fish['이름']} is swimming at {fish['speed']} m/s'''


a = show_fish_movement_Generator(fish_list) # show_fish_movement_Generator(fish_list) 함수를 호출했지만, 이 호출의 결과를 사용하지 않았습니다.
                                            # 단지 함수를 호출한 후 변수 a에 할당하는데 그쳤고, 함수 자체를 출력하거나 다른 방법으로 사용하지는 않았습니다.
                                            # 그런 다음 next(a)를 호출하여 제너레이터의 다음 값을 출력했으므로, 이를 개선하기 위해, 제너레이터 함수를 호출한 후에
                                            # 그 결과를 반복문을 사용하여 출력하는 방식으로 수정할 수 있습니다.

print("Using Comprehension:")
show_fish_movement_comprehension(fish_list)
print("Using Generator:")
show_fish_movement_Generator(fish_list)
print(next(a))   # 이 부분이 두 번 호출됨 (문제 3)
t.sleep(2)       # 문제 1: 't'로 임포트했지만 'time'으로 사용하려 함
print(next(a))   # 문제 3: StopIteration 예외가 발생할 수 있음
```

# 참고 링크 및 코드 개선


> ### 설명:

> 변수명을 영어로 변경하여 국제적으로 이해하기 쉽습니다.
> 제너레이터 함수를 통해 생성된 값을 반복문을 사용하여 출력했고,
> time 모듈 임포트를 상단으로 옮겼습니다.
> 또한 제너레이터 함수를 적절하게 활용하여 필요한 시점에 데이터를 생성하고 출력할 수 있습니다.

```python
import time

fish_list = [
    {"name": "Nemo", "speed": 3},
    {"name": "Dory", "speed": 5},
]


def show_fish_movement_comprehension(fish_list):
    fish_info_list = [(fish['name'], fish['speed']) for fish in fish_list]

    for name, speed in fish_info_list:
        print(f'{name} is swimming at {speed} m/s')
        time.sleep(2)


def show_fish_movement_generator(fish_list):
    for fish in fish_list:
        yield f"{fish['name']} is swimming at {fish['speed']} m/s"
        time.sleep(2)


print("Using Comprehension:")
show_fish_movement_comprehension(fish_list)

print("Using Generator:")
fish_movement_generator = show_fish_movement_generator(fish_list)
for movement_info in fish_movement_generator:
    print(movement_info)
```
