from collections import Counter
from dataclasses import replace
import unittest
from embodiment.scenarios import HELD_OUT, plan, validate_plans, validate_scenario
from dataclasses import asdict
from embodiment.qualify import campaign_cases, campaign_report


class CollectionScenarios(unittest.TestCase):
    def test_balanced_reproducible_heldout_plan(self):
        plans = {split: plan(split, 60) for split in ('train', 'validation', 'evaluation')}
        validate_plans(plans)
        self.assertEqual(plans['train'], plan('train', 60))
        for split, cases in plans.items():
            self.assertEqual(Counter(case.target for case in cases), {'red': 20, 'green': 20, 'blue': 20})
            counts = Counter((case.target, case.region) for case in cases)
            self.assertEqual(len(set(counts.values())), 1)
            self.assertEqual(set(counts) & HELD_OUT, HELD_OUT if split == 'evaluation' else set())
            self.assertTrue(all(case.seed > 19 for case in cases))

    def test_reject_duplicate_fingerprints_and_bad_annotations(self):
        case = plan('train', 1)[0]
        for cases in ([case, case], [replace(case, scenario_id='bad')], [replace(case, region=9)], [replace(case, split='evaluation')]):
            with self.assertRaises(ValueError):
                validate_plans({'train': cases})
        with self.assertRaises(ValueError):
            validate_plans({'evaluation': [replace(case, split='evaluation')]})

    def test_smoke_and_collection_cannot_pass_qualification(self):
        cases = campaign_cases('smoke','train',1)
        trials = [dict(seed=0,success=True,failure_reason=None)]
        report = campaign_report(trials,True,True,None,'smoke',cases,'franka_panda')
        self.assertEqual(report['status'],'COMPLETE')
        self.assertFalse(report['threshold_achieved'])
        cases = campaign_cases('qualification','train',999)
        self.assertEqual([case['seed'] for case in cases],list(range(20)))
        with self.assertRaises(ValueError):
            campaign_cases('smoke','train',20)

    def test_unseen_smoke_seeds_do_not_enter_dataset_or_qualification_schedules(self):
        cases = campaign_cases('smoke', 'train', 3, smoke_seed=1000000000)
        self.assertEqual([case['seed'] for case in cases], [1000000000, 1000000001, 1000000002])
        self.assertTrue(all(case['seed'] < plan('train', 1)[0].seed for case in cases))
        with self.assertRaises(ValueError):
            campaign_cases('qualification', 'train', 20, smoke_seed=1000000000)
        with self.assertRaises(ValueError):
            campaign_cases('smoke', 'train', 1, smoke_seed=-1)

    def test_invalid_requests(self):
        for split, count in (('test', 10), ('train', 0), ('train', True), ('train', 100001)):
            with self.assertRaises(ValueError):
                plan(split, count)

    def test_split_relabeling_and_unassigned_seed_are_rejected(self):
        case = asdict(plan('train', 1)[0])
        validate_scenario(case)
        with self.assertRaisesRegex(ValueError, 'another split'):
            validate_scenario({**case, 'split': 'validation'})
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            validate_scenario({**case, 'seed': case['seed'] + 1})


if __name__ == '__main__':
    unittest.main()
