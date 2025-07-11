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

# 기본 환경 클래스를 임포트합니다. (BaseEnv: 기본 환경 클래스)
from legged_lab.envs.base.base_env import BaseEnv
# 기본 환경 및 에이전트 설정 클래스를 임포트합니다. (BaseAgentCfg: 기본 에이전트 설정, BaseEnvCfg: 기본 환경 설정)
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
# TienKung 로봇의 달리기 설정을 임포트합니다. (TienKungRunAgentCfg: 달리기 에이전트 설정, TienKungRunFlatEnvCfg: 평지 달리기 환경 설정)
from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
# 센서가 포함된 달리기 설정을 임포트합니다. (TienKungRunWithSensorAgentCfg: 센서 포함 달리기 에이전트 설정, TienKungRunWithSensorFlatEnvCfg: 센서 포함 평지 달리기 환경 설정)
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
# TienKung 로봇 환경 클래스를 임포트합니다. (TienKungEnv: TienKung 로봇 환경 클래스)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
# TienKung 로봇의 보행 설정을 임포트합니다. (TienKungWalkAgentCfg: 보행 에이전트 설정, TienKungWalkFlatEnvCfg: 평지 보행 환경 설정)
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
# 센서가 포함된 보행 설정을 임포트합니다. (TienKungWalkWithSensorAgentCfg: 센서 포함 보행 에이전트 설정, TienKungWalkWithSensorFlatEnvCfg: 센서 포함 평지 보행 환경 설정)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
# 태스크 레지스트리 유틸리티를 임포트합니다. (task_registry: 태스크 등록을 위한 객체)
from legged_lab.utils.task_registry import task_registry

# "walk" 태스크를 등록합니다. (TienKungEnv 환경, TienKungWalkFlatEnvCfg 설정, TienKungWalkAgentCfg 에이전트 사용)
task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
# "run" 태스크를 등록합니다. (TienKungEnv 환경, TienKungRunFlatEnvCfg 설정, TienKungRunAgentCfg 에이전트 사용)
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
# "walk_with_sensor" 태스크를 등록합니다. (TienKungEnv 환경, TienKungWalkWithSensorFlatEnvCfg 설정, TienKungWalkWithSensorAgentCfg 에이전트 사용)
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
# "run_with_sensor" 태스크를 등록합니다. (TienKungEnv 환경, TienKungRunWithSensorFlatEnvCfg 설정, TienKungRunWithSensorAgentCfg 에이전트 사용)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)
