# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import math  # 수학 관련 함수를 사용하기 위해 math 모듈을 임포트합니다.

from isaaclab.managers import EventTermCfg as EventTerm     # 이벤트 용어 설정을 위한 클래스 임포트
from isaaclab.managers import RewardTermCfg as RewTerm      # 보상 용어 설정을 위한 클래스 임포트
from isaaclab.managers import SceneEntityCfg                # 장면 엔티티 설정을 위한 클래스 임포트
from isaaclab.utils import configclass                      # 설정 클래스를 정의하기 위한 데코레이터 임포트
from isaaclab_rl.rsl_rl import (                            # noqa:F401
    RslRlOnPolicyRunnerCfg,                                 # RSL-RL의 온-폴리시 러너 설정 클래스
    RslRlPpoActorCriticCfg,                                 # PPO 액터-크리틱 설정 클래스
    RslRlPpoAlgorithmCfg,                                   # PPO 알고리즘 설정 클래스
    RslRlRndCfg,                                            # RND 설정 클래스 (사용되지 않을 수 있음)
    RslRlSymmetryCfg,                                       # 대칭 설정 클래스
)

import legged_lab.mdp as mdp                                    # MDP(마르코프 결정 과정) 관련 함수 모듈 임포트
from legged_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG  # TienKung2 Lite 로봇 설정 임포트
from legged_lab.envs.base.base_config import (                  # 기본 설정 클래스들 임포트
    ActionDelayCfg,                                             # 행동 지연 설정
    BaseSceneCfg,                                               # 기본 장면 설정
    CommandRangesCfg,                                           # 명령 범위 설정
    CommandsCfg,                                                # 명령 설정
    DomainRandCfg,                                              # 도메인 랜덤화 설정
    EventCfg,                                                   # 이벤트 설정
    HeightScannerCfg,                                           # 높이 스캐너 설정
    NoiseCfg,                                                   # 노이즈 설정
    NoiseScalesCfg,                                             # 노이즈 스케일 설정
    NormalizationCfg,                                           # 정규화 설정
    ObsScalesCfg,                                               # 관찰 스케일 설정
    PhysxCfg,                                                   # PhysX 설정
    RobotCfg,                                                   # 로봇 설정
    SimCfg,                                                     # 시뮬레이션 설정
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401  # 지형 설정 임포트 (자갈과 거친 지형)

@configclass                            # 이 클래스를 설정 클래스로 만듭니다.
class GaitCfg:                          # 로봇의 걸음걸이(게이트) 파라미터를 정의하는 클래스
    gait_air_ratio_l: float = 0.38      # 왼쪽 다리의 공중 비율 (발이 땅에 닿지 않는 시간 비율)
    gait_air_ratio_r: float = 0.38      # 오른쪽 다리의 공중 비율
    gait_phase_offset_l: float = 0.38   # 왼쪽 다리의 위상 오프셋 (걸음 주기 내 타이밍 조정)
    gait_phase_offset_r: float = 0.88   # 오른쪽 다리의 위상 오프셋
    gait_cycle: float = 0.85            # 전체 걸음 주기 (초 단위)

####
@configclass  # 이 클래스를 설정 클래스로 만듭니다.
class LiteRewardCfg:  # Lite 모델의 보상 설정 클래스
    # 선형 속도 추적 보상 (xy 평면에서 명령된 속도를 추적, 지수형 페널티)
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    # 각속도 z축 추적 보상 (지수형 페널티)
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    # 수직 선형 속도 L2 페널티 (로봇이 점프하지 않도록)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # xy 각속도 L2 페널티 (회전 안정성)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 에너지 소비 페널티 (효율적 움직임 유도)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    # 관절 가속도 L2 페널티 (부드러운 움직임)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 행동 변화율 L2 페널티 (행동 안정성)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 원치 않는 접촉 페널티 (특정 부위의 충돌 방지)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    # 몸통 방향 L2 페널티 (펠비스 부위의 방향 안정성)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-2.0
    )
    # 평면 방향 L2 페널티 (전체 방향 안정성)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # 종료 페널티 (로봇이 넘어지면 큰 페널티)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # 발 미끄러짐 페널티
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    # 발 힘 페널티 (과도한 힘 방지)
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    # 발이 너무 가까운 페널티 (인간형 로봇의 발 간격 유지)
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.2},
    )
    # 발 걸림 페널티
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )
    # 관절 위치 한계 페널티
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # 엉덩이 관절 편차 페널티 (L1 노름)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_yaw_.*_joint",
                    "hip_roll_.*_joint",
                    "shoulder_pitch_.*_joint",
                    "elbow_pitch_.*_joint",
                ],
            )
        },
    )
    # 팔 관절 편차 페널티
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_roll_.*_joint", "shoulder_yaw_.*_joint"])},
    )
    # 다리 관절 편차 페널티
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_pitch_.*_joint",
                    "knee_pitch_.*_joint",
                    "ankle_pitch_.*_joint",
                    "ankle_roll_.*_joint",
                ],
            )
        },
    )

    # 게이트 발 힘 주기 보상 (주기적인 발 힘 패턴 유도)
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    # 게이트 발 속도 주기 보상
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    # 게이트 발 지지 힘 주기 보상
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})

    # 발목 토크 페널티 (과도한 토크 방지)
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    # 발목 행동 페널티
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    # 엉덩이 롤 행동 페널티
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0)
    # 엉덩이 요 행동 페널티
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)
    # 발 y축 거리 페널티 (발 간격 유지, 하지만 함수가 hip_roll_action으로 오타 가능성)
    feet_y_distance = RewTerm(func=mdp.hip_roll_action, weight=-2.0)
####
####
@configclass  # 이 클래스를 설정 클래스로 만듭니다.
class TienKungWalkFlatEnvCfg:  # TienKung 로봇의 평지 보행 환경 설정 클래스
    amp_motion_files_display = ["legged_lab/envs/tienkung/datasets/motion_visualization/walk.txt"]  # 시각화용 모션 파일
    device: str = "cuda:0"  # 사용 장치 (GPU)
    scene: BaseSceneCfg = BaseSceneCfg(  # 장면 설정
        max_episode_length_s=20.0,  # 최대 에피소드 길이 (20초)
        num_envs=4096,  # 환경 수 (병렬 시뮬레이션)
        env_spacing=2.5,  # 환경 간격
        robot=TIENKUNG2LITE_CFG,  # 로봇 모델
        terrain_type="generator",  # 지형 타입 (생성기)
        terrain_generator=GRAVEL_TERRAINS_CFG,  # 자갈 지형 생성기
        # terrain_type="plane",  # 평면 지형 (주석 처리됨)
        # terrain_generator= None,  # 생성기 없음 (주석 처리됨)
        max_init_terrain_level=5,  # 초기 지형 레벨 최대
        height_scanner=HeightScannerCfg(  # 높이 스캐너 설정
            enable_height_scan=False,  # 높이 스캔 비활성화
            prim_body_name="pelvis",  # 기준 몸통 부위
            resolution=0.1,  # 해상도
            size=(1.6, 1.0),  # 스캔 크기
            debug_vis=False,  # 디버그 시각화 비활성화
            drift_range=(0.0, 0.0),  # 드리프트 범위
        ),
    )
    robot: RobotCfg = RobotCfg(  # 로봇 설정
        actor_obs_history_length=10,  # 액터 관찰 히스토리 길이
        critic_obs_history_length=10,  # 크리틱 관찰 히스토리 길이
        action_scale=0.25,  # 행동 스케일
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],  # 종료 접촉 부위
        feet_body_names=["ankle_roll.*"],  # 발 부위 이름
    )
    reward = LiteRewardCfg()  # 보상 설정 (위 클래스 사용)
    gait = GaitCfg()  # 게이트 설정 (위 클래스 사용)
    normalization: NormalizationCfg = NormalizationCfg(  # 정규화 설정
        obs_scales=ObsScalesCfg(  # 관찰 스케일
            lin_vel=1.0,  # 선형 속도
            ang_vel=1.0,  # 각속도
            projected_gravity=1.0,  # 투영 중력
            commands=1.0,  # 명령
            joint_pos=1.0,  # 관절 위치
            joint_vel=1.0,  # 관절 속도
            actions=1.0,  # 행동
            height_scan=1.0,  # 높이 스캔
        ),
        clip_observations=100.0,  # 관찰 클리핑
        clip_actions=100.0,  # 행동 클리핑
        height_scan_offset=0.5,  # 높이 스캔 오프셋
    )
    commands: CommandsCfg = CommandsCfg(  # 명령 설정
        resampling_time_range=(10.0, 10.0),  # 재샘플링 시간 범위
        # 전체 환경 중 20%는 제자리에 가만히 서 있도록(standing) 명령을 내려라
        rel_standing_envs=0.2,  # 상대 서 있는 환경 비율
        # 나머지 80%의 움직이는 환경 모두에게는 특정 방향(heading)을 바라보도록 명령을 내려라
        rel_heading_envs=1.0,  # 상대 헤딩 환경 비율
        heading_command=True,  # 헤딩 명령 활성화
        heading_control_stiffness=0.5,  # 헤딩 제어 강성
        debug_vis=True,  # 디버그 시각화
        ranges=CommandRangesCfg(  # 명령 범위
            lin_vel_x=(-0.6, 1.0),  # x 선형 속도
            lin_vel_y=(-0.5, 0.5),  # y 선형 속도
            ang_vel_z=(-1.57, 1.57),  # z 각속도
            heading=(-math.pi, math.pi)  # 헤딩
        ),
    )
    noise: NoiseCfg = NoiseCfg(  # 노이즈 설정
        add_noise=True,  # 노이즈 추가 활성화
        noise_scales=NoiseScalesCfg(  # 노이즈 스케일
            lin_vel=0.2,  # 선형 속도 노이즈
            ang_vel=0.2,  # 각속도 노이즈
            projected_gravity=0.05,  # 투영 중력 노이즈
            joint_pos=0.01,  # 관절 위치 노이즈
            joint_vel=1.5,  # 관절 속도 노이즈
            height_scan=0.1,  # 높이 스캔 노이즈
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(  # 도메인 랜덤화 설정
        events=EventCfg(  # 이벤트 설정
            physics_material=EventTerm(  # 물리 재질 랜덤화
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(  # 기본 질량 추가 랜덤화
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(  # 기본 상태 리셋
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(  # 로봇 관절 리셋
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(  # 로봇 푸시 이벤트
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),  # 행동 지연 설정 (비활성화)
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))  # 시뮬레이션 설정 (타임스텝, 디시메이션, PhysX)

####

####
@configclass  # 이 클래스를 설정 클래스로 만듭니다.
class TienKungWalkAgentCfg(RslRlOnPolicyRunnerCfg):  # TienKung 보행 에이전트 설정 클래스 (RSL-RL 온-폴리시 러너 상속)
    seed = 42  # 시드 값 (재현성)
    device = "cuda:0"  # 장치
    num_steps_per_env = 24  # 환경당 스텝 수
    max_iterations = 50000  # 최대 반복 횟수
    empirical_normalization = False  # 경험적 정규화 비활성화
    policy = RslRlPpoActorCriticCfg(  # 정책 설정 (액터-크리틱)
        class_name="ActorCritic",
        init_noise_std=1.0,  # 초기 노이즈 표준편차
        noise_std_type="scalar",  # 노이즈 타입
        actor_hidden_dims=[512, 256, 128],  # 액터 히든 레이어 크기
        critic_hidden_dims=[512, 256, 128],  # 크리틱 히든 레이어 크기
        activation="elu",  # 활성화 함수
    )
    algorithm = RslRlPpoAlgorithmCfg(  # 알고리즘 설정 (AMPPPO)
        class_name="AMPPPO",
        value_loss_coef=1.0,  # 가치 손실 계수
        use_clipped_value_loss=True,  # 클리핑 가치 손실 사용
        clip_param=0.2,  # 클립 파라미터
        entropy_coef=0.005,  # 엔트로피 계수
        num_learning_epochs=5,  # 학습 에포크 수
        num_mini_batches=4,  # 미니 배치 수
        learning_rate=1.0e-3,  # 학습률
        schedule="adaptive",  # 스케줄러
        gamma=0.99,  # 할인율
        lam=0.95,  # 람다 (GAE)
        desired_kl=0.01,  # 원하는 KL 발산
        max_grad_norm=1.0,  # 최대 그래디언트 노름
        normalize_advantage_per_mini_batch=False,  # 미니 배치별 이점 정규화 비활성화
        symmetry_cfg=None,  # 대칭 설정 (None)
        rnd_cfg=None,  # RND 설정 (None)
    )
    clip_actions = None  # 행동 클리핑 (None)
    save_interval = 100  # 저장 간격
    runner_class_name = "AmpOnPolicyRunner"  # 러너 클래스 이름
    experiment_name = "walk"  # 실험 이름
    run_name = ""  # 런 이름
    logger = "tensorboard"  # 로거 (Tensorboard)
    neptune_project = "walk"  # Neptune 프로젝트
    wandb_project = "walk"  # WandB 프로젝트
    resume = False  # 재개 여부
    load_run = ".*"  # 로드 런
    load_checkpoint = "model_.*.pt"  # 로드 체크포인트

    # AMP 파라미터 (Adversarial Motion Prior)
    amp_reward_coef = 0.3  # AMP 보상 계수
    amp_motion_files = ["legged_lab/envs/tienkung/datasets/motion_amp_expert/walk.txt"]  # AMP 모션 파일
    amp_num_preload_transitions = 200000  # 사전 로드 전이 수
    amp_task_reward_lerp = 0.7  # 태스크 보상 선형 보간
    amp_discr_hidden_dims = [1024, 512, 256]  # 판별기 히든 레이어 크기
    min_normalized_std = [0.05] * 20  # 최소 정규화 표준편차 (20개 관절)
####