from dataclasses import replace
import unittest
from embodiment.benchmark import Action, BIN, COLORS, Observation, layout
from embodiment.execution import ClosedLoop, ExecutionConfig
from embodiment.safety import SafetyConfig


class Backend:
    embodiment = 'openmanipulator_x'
    safety_config = SafetyConfig(((-3.,3.),)*4+((-.011,.02),)*2,4)
    def __init__(self): self.commands = []; self.stopped = False
    def step(self, action): self.commands.append(action); return action
    def safe_stop(self): self.stopped = True


class MockPolicy:
    def __init__(self,error=False): self.error = error; self.inputs = []
    def reset(self): self.reset_called = True
    def actions(self,obs):
        self.inputs.append(obs)
        if self.error: raise ValueError('mock invalid model output')
        return (Action((.001,0,0),1.),)*16


def observation(time=0.):
    return Observation(time,(.17,0.,.085),(0.,)*6,(0.,)*6,
                       {c:(*p,1.,0.,0.,0.) for c,p in layout(0).items()},
                       {c:0. for c in COLORS},gripper_open=1.)


class Execution(unittest.TestCase):
    def test_chunks_replan_without_privileged_input(self):
        backend,policy = Backend(),MockPolicy()
        executor = ClosedLoop(backend,policy,'red',ExecutionConfig(execution_horizon=1))
        for tick in range(2): executor.tick(observation(tick/30),None)
        self.assertEqual(len(policy.inputs),2)
        self.assertFalse(hasattr(policy.inputs[0],'cubes'))
        self.assertEqual(len(backend.commands),2)
        self.assertEqual(len(executor.protocol_digest()),64)

    def test_policy_failure_uses_classical_fallback_and_retains_attribution(self):
        executor = ClosedLoop(Backend(),MockPolicy(True),'red')
        executor.tick(observation(),None)
        self.assertIsNotNone(executor.expert)
        executor.tick(observation(45.),None)
        self.assertEqual(executor.result()['controller'],'fallback_assisted')
        self.assertFalse(executor.result()['success'])

    def test_disabled_fallback_and_unsafe_state_stop(self):
        for unsafe in (False,True):
            backend = Backend()
            executor = ClosedLoop(backend,MockPolicy(True),'red',ExecutionConfig(fallback_enabled=False))
            executor.tick(replace(observation(),unsafe=unsafe),None)
            self.assertTrue(backend.stopped)
            self.assertEqual(len(backend.commands),0)
            self.assertEqual(executor.result()['fallback_triggers'],[])

    def test_unsafe_state_never_enters_enabled_fallback(self):
        backend,policy = Backend(),MockPolicy()
        executor = ClosedLoop(backend,policy,'red')
        executor.tick(replace(observation(),unsafe=True),None)
        self.assertTrue(backend.stopped)
        self.assertEqual(policy.inputs,[])
        self.assertEqual(backend.commands,[])
        self.assertIsNone(executor.expert)
        self.assertEqual(executor.result()['failure_reason'],'unsafe_movement')

    def test_actual_lift_release_and_dwell_required_for_success(self):
        executor = ClosedLoop(Backend(),MockPolicy(),'red')
        first = observation()
        first.cubes['red'] = (.17,0.,.07,1.,0.,0.,0.)
        first.contacts['red'] = True
        executor.tick(first,None)
        placed = observation(.1)
        placed.cubes['red'] = (*BIN[:2],BIN[2]+.0125,1.,0.,0.,0.)
        executor.tick(placed,None)
        executor.tick(replace(placed,time=1.2),None)
        self.assertTrue(executor.result()['success'])
        self.assertEqual(executor.result()['controller'],'vla_only')

    def test_wrong_color_is_never_recovered_as_policy_success(self):
        executor = ClosedLoop(Backend(),MockPolicy(),'red')
        obs = observation(); obs.cubes['blue'] = (.17,0.,.06,1.,0.,0.,0.)
        executor.tick(obs,None)
        self.assertEqual(executor.result()['failure_reason'],'wrong_color')

    def test_timeout_is_accounted_with_and_without_fallback(self):
        class TimeoutPolicy(MockPolicy):
            def actions(self, obs):
                raise TimeoutError('unit fixture timeout')
        for enabled in (False, True):
            backend = Backend()
            executor = ClosedLoop(backend, TimeoutPolicy(), 'red', ExecutionConfig(fallback_enabled=enabled))
            executor.tick(observation(), None)
            if enabled:
                executor.tick(observation(45.), None)
            result = executor.result()
            self.assertEqual(result['policy_errors'], ['TimeoutError'])
            self.assertEqual(result['controller'], 'fallback_assisted' if enabled else 'vla_only')

    def test_rejected_unsafe_policy_command_is_retained_after_fallback(self):
        class UnsafePolicy(MockPolicy):
            def actions(self, obs):
                return (Action((.003, 0., 0.), 1.),)
        executor = ClosedLoop(Backend(), UnsafePolicy(), 'red')
        executor.tick(replace(observation(), ee=(.281, 0., .085)), None)
        executor.tick(observation(45.), None)
        result = executor.result()
        self.assertEqual(result['rejected_policy_actions'], ['workspace_command'])
        self.assertEqual(result['fallback_triggers'], ['unsafe_policy_action:workspace_command'])
        self.assertEqual(result['safety_events'], [])

    def test_simulator_stall_stops_execution(self):
        backend = Backend()
        executor = ClosedLoop(backend, MockPolicy(), 'red')
        executor.tick(observation(), None)
        executor.tick(observation(), None)
        self.assertTrue(backend.stopped)
        self.assertEqual(len(backend.commands), 1)
        self.assertEqual(executor.result()['failure_reason'], 'unsafe_movement')
