import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import random
import warnings
import platform

from numpy import floating

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def setup_korean_font():
    """한글 폰트 설정"""
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False


setup_korean_font()

def generate_initial_occupancy(num_floors: int,
                               occupancy_pattern: str = 'linear',
                               overall_fullness: float = 0.5) -> np.ndarray:
    """
    Parameters:
    - occupancy_pattern: 'linear', 'exponential', 'sigmoid' 등
    - overall_fullness: 전체 평균 점유율 (0~1)

    Returns:
    - 각 층의 점유율 배열
    """
    floors = np.arange(1, num_floors + 1)

    if occupancy_pattern == 'linear':
        # 선형 감소: 1층이 가장 높음
        rates = 1 - (floors - 1) / num_floors

    elif occupancy_pattern == 'exponential':
        # 지수적 감소
        rates = np.exp(-0.3 * (floors - 1) / num_floors)

    elif occupancy_pattern == 'sigmoid':
        # S자 곡선 (중간층이 급격히 변화)
        x = (floors - num_floors / 2) / (num_floors / 4)
        rates = 1 / (1 + np.exp(x))

    elif occupancy_pattern == 'uniform':
        # 균등 분포
        rates = np.ones(num_floors)

    # overall_fullness에 맞게 스케일 조정
    rates = rates / rates.mean() * overall_fullness
    rates = np.clip(rates, 0, 1)  # 0~1 범위로 제한

    return rates

class ParkingGarage:
    """엔트로피 기반 주차장 시뮬레이터"""

    def __init__(self, num_floors: int = 10, spots_per_floor: int = 30,
                 t1: float = 1.0, t2: float = 0.5, t3: float = 0.3,
                 initial_temp: float = 0.5, k_B: float = 1.0,
                 init_occupancy_rates: List[float] = None,
                 occupancy_pattern: str = 'linear',
                 overall_fullness: float = 0.7):
        """

        Parameters:
        - num_floors: 층 수
        - spots_per_floor: 층당 주차 공간 수
        - t1: 한 층을 스캔하는 시간
        - t2: 도보로 한 층 올라가는 시간
        - t3: 차량으로 한 층 내려가는 시간
        - initial_temp: 초기 온도 추정값 (첫 업데이트 전까지만 사용)
        - k_B: 볼츠만 상수
        - init_occupancy_rates: 직접 지정 (우선순위 높음)
        - occupancy_pattern: 패턴 ('linear', 'exponential', 'sigmoid', 'uniform')
        - overall_fullness: 전체 평균 점유율 (0~1)
        """
        self.num_floors = num_floors
        self.spots_per_floor = spots_per_floor
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.T = initial_temp
        self.k_B = k_B

        # 각 층의 에너지 (거리의 제곱)
        self.energies = np.array([(i / num_floors) ** 2
                                  for i in range(1, num_floors + 1)])

        # 주차 상태 (0: 빈 공간, 1: 차 있음)
        self.parking_state = np.zeros((num_floors, spots_per_floor), dtype=int)

        # 통계
        self.history = []
        self.temperature_history= []

        # 초기 분포 생성
        if init_occupancy_rates is None:
            init_occupancy_rates = generate_initial_occupancy(
                num_floors, occupancy_pattern, overall_fullness
            )

        self._initialize_with_rates(init_occupancy_rates)
        self.update_temperature()

        print(f"=== 초기화 완료 ===")
        print(f"초기 분포 → 추정 온도: {self.T:.3f}\n")


    def _initialize_with_rates(self, occupancy_rates: np.ndarray):
        """주어진 점유율로 주차장 초기화 (T와 무관)"""
        print(f"초기 점유율 설정:")
        for floor_idx, rate in enumerate(occupancy_rates):
            num_occupied = int(rate * self.spots_per_floor)
            if num_occupied > 0:
                spots = np.random.choice(
                    self.spots_per_floor,
                    size=num_occupied,
                    replace=False
                )
                self.parking_state[floor_idx, spots] = 1
            print(f"  {floor_idx + 1}층: {rate * 100:.1f}%")

    def calculate_occupancy_probabilities(self) -> np.ndarray:
        """현재 온도에서 각 층의 점유 확률 예측"""
        probabilities = []
        for E in self.energies:
            exp_term = np.exp(-E / (self.k_B * self.T))
            p = 2 * exp_term / (1 + exp_term)
            probabilities.append(p)
        return np.array(probabilities)

    def dynamic_programming(self, probabilities: np.ndarray) -> Dict[int, int]:
        """
        동적 계획법으로 최적 전략 계산

        Returns:
        - strategy: {현재_층: 실패시_다음_층} 딕셔너리
        """
        # f[i]: i층에서 시작할 때 최소 예상 시간
        f = np.zeros(self.num_floors + 1)
        strategy = {}

        # 경계 조건: N층 (무조건 주차)
        f[self.num_floors] = self.t1 + self.num_floors * self.t2

        # 역순 계산 (N-1층 → 1층)
        for i in range(self.num_floors - 1, 0, -1):
            pi = probabilities[i - 1]  # i층 점유 확률

            # 성공: i층에서 주차
            success_time = self.t1 + i * self.t2

            # 실패: 다음 층 찾기
            best_next_time = float('inf')
            best_next_floor = self.num_floors

            for j in range(i + 1, self.num_floors + 1):
                move_time = (j - i) * self.t3
                total_time = move_time + f[j]

                if total_time < best_next_time:
                    best_next_time = total_time
                    best_next_floor = j

            # 기댓값: pi*성공 + (1-pi)*(t1+실패)
            f[i] = pi * success_time + (1 - pi) * (self.t1 + best_next_time)
            strategy[i] = best_next_floor

        # 0층 (입구)
        best_time = float('inf')
        best_floor = 1

        for j in range(1, self.num_floors + 1):
            time = j * self.t3 + f[j]
            if time < best_time:
                best_time = time
                best_floor = j

        f[0] = best_time
        strategy[0] = best_floor

        return strategy

    def try_park_on_floor(self, floor: int) -> bool:
        """특정 층에서 주차 시도"""
        empty_spots = np.where(self.parking_state[floor - 1] == 0)[0]

        if len(empty_spots) > 0:
            spot = random.choice(empty_spots)
            self.parking_state[floor - 1][spot] = 1
            return True
        return False

    def get_actual_occupancy_rate(self, floor: int) -> float:
        """특정 층의 실제 점유율"""
        occupied = np.sum(self.parking_state[floor - 1])
        return occupied / self.spots_per_floor

    def update_temperature(self):
        """실제 분포를 관찰해서 T 추정 (국소 그리드 서치)"""
        actual_rates = np.array([
            self.get_actual_occupancy_rate(floor)
            for floor in range(1, self.num_floors + 1)
        ])

        best_T = self.T
        best_mse = self._compute_mse(self.T, actual_rates)

        # T 주변 ±0.2 범위 탐색
        search_range = 0.2
        T_min = max(0.1, self.T - search_range)  # 최소 0.1
        T_max = min(2.0, self.T + search_range)

        T_candidates = np.linspace(T_min, T_max, 41)

        for T_candidate in T_candidates:
            mse = self._compute_mse(T_candidate, actual_rates)
            if mse < best_mse:
                best_mse = mse
                best_T = T_candidate

        self.T = best_T
        self.temperature_history.append(self.T)

    def _compute_mse(self, T: float, actual_rates: np.ndarray) -> floating[Any]:
        """특정 온도에서 MSE 계산"""
        predicted = np.array([
            2 * np.exp(-E / (self.k_B * T)) / (1 + np.exp(-E / (self.k_B * T)))
            for E in self.energies
        ])
        return np.mean((predicted - actual_rates) ** 2)

    def park_car_with_strategy(self, strategy: Dict[int, int],
                               probabilities: np.ndarray,
                               car_id: int) -> Tuple[int, float]:
        """전략에 따라 차량 주차 (예측 확률 사용)"""
        current_floor = 0
        total_time = 0
        scan_count = 0
        path = [0]

        while current_floor <= self.num_floors:
            # 다음 목표 층
            target_floor = strategy.get(current_floor, self.num_floors)

            # 이동
            if current_floor == 0:
                move_time = target_floor * self.t3
            else:
                move_time = (target_floor - current_floor) * self.t3

            total_time += move_time
            current_floor = target_floor
            path.append(current_floor)

            # 스캔
            total_time += self.t1
            scan_count += 1

            # 주차 시도 (예측 확률 사용)
            pi = probabilities[current_floor - 1]

            # (1-pi) 확률로 빈 공간
            if random.random() < (1 - pi):
                if self.try_park_on_floor(current_floor):
                    total_time += current_floor * self.t2
                    self.history.append({
                        'car_id': car_id,
                        'floor': current_floor,
                        'time': total_time,
                        'scans': scan_count,
                        'path': path,
                        'temperature': self.T
                    })
                    return current_floor, total_time

            # 최하층이면 강제 주차
            if current_floor >= self.num_floors:
                if not self.try_park_on_floor(current_floor):
                    # 꽉 찼으면 빈 층 찾기
                    for f in range(self.num_floors, 0, -1):
                        if self.try_park_on_floor(f):
                            current_floor = f
                            break

                total_time += current_floor * self.t2
                self.history.append({
                    'car_id': car_id,
                    'floor': current_floor,
                    'time': total_time,
                    'scans': scan_count,
                    'path': path,
                    'temperature': self.T
                })
                return current_floor, total_time

        return current_floor, total_time

    def benchmark_policy(self, car_id: int) -> Tuple[int, float]:
        """벤치마크 정책: 1층부터 순차적으로"""
        total_time = 0

        for floor in range(1, self.num_floors + 1):
            total_time += self.t3  # 이동
            total_time += self.t1  # 스캔

            # 실제 점유율로 주차 시도
            actual_rate = self.get_actual_occupancy_rate(floor)
            if actual_rate < 1.0:
                if random.random() < (1 - actual_rate):
                    self.try_park_on_floor(floor)
                    total_time += floor * self.t2
                    return floor, total_time

        # 최하층 강제
        self.try_park_on_floor(self.num_floors)
        total_time += self.num_floors * self.t2
        return self.num_floors, total_time

    def simulate(self, num_cars: int = 30, update_temp_interval: int = 5):
        """시뮬레이션 실행"""
        print(f"=== TIPP 시뮬레이션 시작 ===")
        print(f"층 수: {self.num_floors}, 층당 주차 공간: {self.spots_per_floor}\n")

        for car_id in range(1, num_cars + 1):
            # 1. 현재 T로 확률 예측
            probabilities = self.calculate_occupancy_probabilities()

            # 2. DP로 전략 계산
            strategy = self.dynamic_programming(probabilities)

            # 3. 주차
            floor, time = self.park_car_with_strategy(strategy, probabilities, car_id)

            print(f"차량 {car_id:2d}: {floor:2d}층, 시간: {time:5.2f}, T: {self.T:.3f}")

            # 4. 주기적으로 T 업데이트
            if car_id % update_temp_interval == 0:
                self.update_temperature()
                print(f"  → T 업데이트: {self.T:.3f}")

        print(f"\n총 {num_cars}대 주차 완료!")
        print(f"최종 온도: {self.T:.3f}")
        print(f"평균 주차 시간: {np.mean([h['time'] for h in self.history]):.2f}")

    def visualize_results(self):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 주차 시간
        ax1 = axes[0, 0]
        times = [h['time'] for h in self.history]
        car_ids = [h['car_id'] for h in self.history]
        ax1.plot(car_ids, times, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('차량 번호', fontsize=12)
        ax1.set_ylabel('주차 시간', fontsize=12)
        ax1.set_title('차량별 주차 소요 시간', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. 온도 변화
        ax2 = axes[0, 1]
        ax2.plot(self.temperature_history, 'r-o', linewidth=2, markersize=4)
        ax2.set_xlabel('업데이트 횟수', fontsize=12)
        ax2.set_ylabel('온도 (T)', fontsize=12)
        ax2.set_title('온도 변화 추이', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. 층별 분포
        ax3 = axes[1, 0]
        floors = [h['floor'] for h in self.history]
        floor_counts = [floors.count(i) for i in range(1, self.num_floors + 1)]
        ax3.bar(range(1, self.num_floors + 1), floor_counts,
                color='steelblue', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('층', fontsize=12)
        ax3.set_ylabel('주차 차량 수', fontsize=12)
        ax3.set_title('층별 주차 분포', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 히트맵
        ax4 = axes[1, 1]
        im = ax4.imshow(self.parking_state, cmap='RdYlGn_r', aspect='auto')
        ax4.set_xlabel('주차 공간', fontsize=12)
        ax4.set_ylabel('층', fontsize=12)
        ax4.set_title('최종 주차장 상태', fontsize=14, fontweight='bold')
        ax4.set_yticks(range(self.num_floors))
        ax4.set_yticklabels(range(1, self.num_floors + 1))
        plt.colorbar(im, ax=ax4)

        plt.tight_layout()
        plt.savefig('parking_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n=== 상세 통계 ===")
        print(f"평균: {np.mean(times):.2f}")
        print(f"중앙값: {np.median(times):.2f}")
        print(f"표준편차: {np.std(times):.2f}")
        print(f"\n층별 점유율:")
        for floor in range(1, self.num_floors + 1):
            rate = self.get_actual_occupancy_rate(floor)
            print(f"  {floor:2d}층: {rate * 100:5.1f}%")


class PolicyComparison:
    """정책 비교"""

    def __init__(self, num_floors: int = 10, spots_per_floor: int = 30,
                 initial_temp: float = 0.5,
                 init_occupancy_rates: List[float] = None,
                 occupancy_pattern: str = 'linear',
                 overall_fullness: float = 0.7):
        self.num_floors = num_floors
        self.spots_per_floor = spots_per_floor
        self.initial_temp = initial_temp
        self.init_occupancy_rates = init_occupancy_rates
        self.occupancy_pattern = occupancy_pattern
        self.overall_fullness = overall_fullness

    def compare_policies(self, num_cars: int = 30, num_runs: int = 5):
        """여러 정책 성능 비교"""
        results = {'TIPP': [], 'Benchmark': []}

        print("=== 정책 비교 시뮬레이션 ===\n")

        for run in range(num_runs):
            print(f"실행 {run + 1}/{num_runs}...")

            if self.init_occupancy_rates is None:
                common_rates = generate_initial_occupancy(
                    self.num_floors,
                    self.occupancy_pattern,
                    self.overall_fullness
                )
            else:
                common_rates = self.init_occupancy_rates


            # TIPP
            garage_tipp = ParkingGarage(
                num_floors=self.num_floors,
                spots_per_floor=self.spots_per_floor,
                initial_temp=self.initial_temp,
                init_occupancy_rates=common_rates,
                occupancy_pattern=self.occupancy_pattern,
                overall_fullness=self.overall_fullness
            )
            for car_id in range(1, num_cars + 1):
                probs = garage_tipp.calculate_occupancy_probabilities()
                strategy = garage_tipp.dynamic_programming(probs)
                _, time = garage_tipp.park_car_with_strategy(strategy, probs, car_id)
                results['TIPP'].append(time)
                if car_id % 5 == 0:
                    garage_tipp.update_temperature()

            # Benchmark
            garage_bench = ParkingGarage(
                num_floors=self.num_floors,
                spots_per_floor=self.spots_per_floor,
                initial_temp=self.initial_temp,
                init_occupancy_rates=common_rates,
                occupancy_pattern=self.occupancy_pattern,
                overall_fullness=self.overall_fullness
            )
            for car_id in range(1, num_cars + 1):
                _, time = garage_bench.benchmark_policy(car_id)
                results['Benchmark'].append(time)

        self.visualize_comparison(results, num_cars)

    def visualize_comparison(self, results: Dict[str, List[float]], num_cars: int):
        """비교 결과 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 누적 시간
        ax1 = axes[0]
        for policy, times in results.items():
            grouped = [times[i:i + num_cars] for i in range(0, len(times), num_cars)]
            avg_cumulative = np.mean([np.cumsum(t) for t in grouped], axis=0)
            ax1.plot(range(1, num_cars + 1), avg_cumulative,
                     linewidth=2, marker='o', markersize=4, label=policy)

        ax1.set_xlabel('차량 수', fontsize=12)
        ax1.set_ylabel('누적 주차 시간', fontsize=12)
        ax1.set_title('정책별 누적 시간 비교', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 박스플롯
        ax2 = axes[1]
        data = [results[p] for p in ['TIPP', 'Benchmark']]
        bp = ax2.boxplot(data, labels=['TIPP', 'Benchmark'], patch_artist=True)

        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
            patch.set_facecolor(color)

        ax2.set_ylabel('주차 시간', fontsize=12)
        ax2.set_title('정책별 주차 시간 분포', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('policy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n=== 정책별 평균 주차 시간 ===")
        for policy, times in results.items():
            print(f"{policy:12s}: {np.mean(times):6.2f} ± {np.std(times):5.2f}")


# ==================== 실행 ====================

if __name__ == "__main__":
    # ===== 설정 =====
    # 시나리오 파라미터
    SCENARIO_CONFIG = {
        'pattern': 'linear',  # 'linear', 'exponential', 'sigmoid', 'uniform'
        'fullness': 0.7,  # 전체 평균 점유율 (0~1)
        'custom_rates': None  # 또는 직접 지정: [0.95, 0.90, ...]
    }

    # 주차장 파라미터
    GARAGE_CONFIG = {
        'num_floors': 10,
        'spots_per_floor': 30,
        't1': 1.0,
        't2': 0.5,
        't3': 0.3,
        'initial_temp': 0.5
    }

    # 시뮬레이션 파라미터
    SIM_CONFIG = {
        'num_cars': 30,
        'update_temp_interval': 1
    }

    # 비교 파라미터
    COMPARISON_CONFIG = {
        'num_cars': 30,
        'num_runs': 1
    }

    # ===== 실행 =====
    print("=" * 50)
    print(f"시나리오: {SCENARIO_CONFIG['pattern']} 분포, "
          f"평균 {SCENARIO_CONFIG['fullness'] * 100:.0f}% 점유")
    print("=" * 50)

    # TIPP 시뮬레이션
    garage = ParkingGarage(
        **GARAGE_CONFIG,
        init_occupancy_rates=SCENARIO_CONFIG['custom_rates'],
        occupancy_pattern=SCENARIO_CONFIG['pattern'],
        overall_fullness=SCENARIO_CONFIG['fullness']
    )

    garage.simulate(**SIM_CONFIG)
    garage.visualize_results()

    # 정책 비교
    print("\n" + "=" * 50)
    print("정책 비교")
    print("=" * 50)

    comparison = PolicyComparison(
        num_floors=GARAGE_CONFIG['num_floors'],
        spots_per_floor=GARAGE_CONFIG['spots_per_floor'],
        initial_temp=GARAGE_CONFIG['initial_temp'],
        init_occupancy_rates=SCENARIO_CONFIG['custom_rates'],
        occupancy_pattern=SCENARIO_CONFIG['pattern'],
        overall_fullness=SCENARIO_CONFIG['fullness']
    )

    comparison.compare_policies(**COMPARISON_CONFIG)