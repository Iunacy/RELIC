import os
from pathlib import Path
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('RELIC_GROOT_SOURCE'), 'set RELIC_GROOT_SOURCE for real upstream trainer mechanics')
class UpstreamOptimizerRecovery(unittest.TestCase):
    def test_cpu_linear_fixture_restores_real_optimizer_scheduler_and_rng(self):
        from embodiment.upstream_validation import verify_source
        source = Path(os.environ['RELIC_GROOT_SOURCE'])
        verify_source(source)
        sys.path.insert(0, str(source))
        import torch
        from transformers import TrainingArguments, TrainerCallback
        from gr00t.experiment.trainer import Gr00tTrainer

        class Dataset(torch.utils.data.Dataset):
            seed = 42
            def __len__(self): return 4
            def __getitem__(self, index): return {'value': torch.tensor([1., 2.])}
            def reset_seed(self, value): self.seed = value

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                with torch.no_grad():
                    self.linear.weight.fill_(.25)
                    self.linear.bias.zero_()
            def forward(self, value): return {'loss': (self.linear(value)-1).square().mean()}

        class Interrupt(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_training_stop = True
                    control.should_save = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def trainer(path, interrupted=False):
                args = TrainingArguments(output_dir=str(path), max_steps=2, per_device_train_batch_size=1,
                    save_steps=1, learning_rate=.01, use_cpu=True, report_to='none', disable_tqdm=True,
                    dataloader_num_workers=0, dataloader_pin_memory=False)
                return Gr00tTrainer(model=Model(), args=args, train_dataset=Dataset(),
                                    callbacks=[Interrupt()] if interrupted else [])
            first = trainer(root/'resumed', True)
            first.train()
            self.assertEqual(first.state.global_step, 1)
            checkpoint = root/'resumed/checkpoint-1'
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                self.assertTrue((checkpoint/name).is_file())
            resumed = trainer(root/'resumed')
            resumed.train(resume_from_checkpoint=True)
            self.assertEqual(resumed.state.global_step, 2)
            self.assertEqual(resumed.train_dataset.seed, 43)
            uninterrupted = trainer(root/'uninterrupted')
            uninterrupted.train()
            for actual, expected in zip(resumed.model.parameters(), uninterrupted.model.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            state = torch.load(root/'resumed/checkpoint-2/optimizer.pt', map_location='cpu', weights_only=True)
            self.assertTrue(all(int(item['step']) == 2 for item in state['state'].values()))
