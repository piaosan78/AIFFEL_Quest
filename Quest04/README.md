# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 박영준
- 리뷰어 : 김건


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [ ] 주석을 보고 작성자의 코드가 이해되었나요?
  > 코드에 이해하기 어려운 부분들이 있었지만 코드 하나하나에 주석 설명이 붙어서 이해하기가 쉬웠습니다. 
- [ ] 코드가 에러를 유발할 가능성이 없나요?
  > 없었습니다.
- [ ] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네 제너레이터와 컴프리헨션을 사용하여 물고기의 움직임을 출력하는 함수를 만든것을 보니 제대로 이해했습니다.
- [ ] 코드가 간결한가요?
  > 비교대상이 없어서 간결한지는 잘 모르겠으나 chat gpt를 통해 똑같은 문제를 물어본 결과 간결한것같습니다

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.

# 물고기 리스트를 생성한다.

fish_list = [
    {'이름': 'Nemo', 'speed': 3},
    {'이름': 'Dory', 'speed': 5},
]
물고기들의 정보를 담고 있는 리스트를 생성한다.

# 물고기 리스트생성

def show_fish_movement_comprehension(fish_list):
  #fish list에 있는  키, 밸류 값을 튜플 형식으로 저장
    출력값 = [(fish['이름'],fish['speed']) for fish in fish_list]
    #출력값에서 name speed 의 변수로 for 반복문
    for name,speed in 출력값:
      #각 반복되는 변수들을 밑에 형식으로 출력
      print(name,'is swimming at',speed, 'm/s')
      #프린트를 한 후 2초를 정지
      t.sleep(2)

# 제너레이터를 사용값을 출력한다.

def show_fish_movement_Generator(fish_list):
  #newlist로 입력 변수값 저장
    newlist = fish_list
    #yield값을 얻어낼 generator 함수
    def generator(newlist):
        for fish in newlist:
            yield f'''{fish['이름']} is swimming at {fish['speed']} m/s'''
    #generator 진행상황을 위해 generator 값을 a로 지정
    a = generator(newlist)
    #generator가 진행 되는 동한 값을 출력
    for i in generator(newlist):
        print(next(a))
        t.sleep(2)
        

# 출력

```print("Using Comprehension:")
show_fish_movement_comprehension(fish_list)
print("Using Generator:")
show_fish_movement_Generator(fish_list)



