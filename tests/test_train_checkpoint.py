import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader

from src.runner.train import Trainer, TrainConfig, replace_checkpoint


@contextlib.contextmanager
def project_temporary_directory():
    root = Path(__file__).resolve().parent
    temporary = tempfile.TemporaryDirectory(dir=root)
    target = Path(temporary.name).resolve()
    assert target.is_relative_to(root)
    try:
        yield str(target)
    finally:
        temporary.cleanup()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)
        self.dropout = torch.nn.Dropout(0.2)
        self.config = SimpleNamespace(id2label={0: 'a', 1: 'b'})

    def forward(self, inputs, labels):
        logits = self.layer(self.dropout(inputs))
        return SimpleNamespace(
            logits=logits,
            loss=torch.nn.functional.cross_entropy(logits, labels),
        )


class Progress:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def set_postfix(self, **kwargs):
        pass


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.enterContext(patch('src.runner.train.SummaryWriter', return_value=Mock()))
        self.enterContext(patch('src.runner.train.tqdm', side_effect=lambda data, **kw: Progress(data)))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_early_stopping_mid_epoch_and_resume(self):
        with project_temporary_directory() as folder:
            root = Path(folder)
            runner = self.make_runner(root)
            runner.training_config.eval_steps = 10
            runner.training_config.early_stopping_patience = 2
            runner.train()
            self.assertEqual(runner.step, 30)
            self.assertEqual(runner.evaluate.call_count, 3)
            runner.save_model.assert_any_call('best')
            runner.save_model.assert_called_with()
            resumed = self.make_runner(root)
            resumed.training_config.early_stopping_patience = 2
            self.assertEqual(resumed.early_stopping_bad_evals, 2)
            self.assertEqual(resumed.early_stopping_best_loss, 0.5)
            self.assertEqual(resumed.next_batch, 30)
            resumed.train()
            self.assertEqual(resumed.step, 30)
            resumed.evaluate.assert_not_called()

    def test_early_stopping_epoch_end_and_disabled(self):
        with project_temporary_directory() as folder:
            for patience, expected_step in [(1, 210), (0, 315)]:
                runner = self.make_runner(Path(folder) / str(patience))
                runner.training_config.epochs = 3
                runner.training_config.eval_steps = 1000
                runner.training_config.early_stopping_patience = patience
                runner.train()
                self.assertEqual(runner.step, expected_step)

    def test_min_delta_reset_duplicate_and_checkpoint_continuity(self):
        with project_temporary_directory() as folder:
            root = Path(folder)
            runner = self.make_runner(root)
            runner.training_config.early_stopping_min_delta = 0.1
            for step, loss, expected_bad in [(1, 1.0, 0), (2, 0.96, 1), (3, 0.85, 0), (4, 0.84, 1)]:
                runner.step = step
                runner.evaluate.return_value['loss'] = loss
                runner.validate_and_save_best()
                runner.validate_and_save_best()
                self.assertEqual(runner.early_stopping_bad_evals, expected_bad)
            self.assertEqual(runner.evaluate.call_count, 4)
            self.assertEqual(runner.min_loss, 0.84)
            runner.save_checkpoint(0, 4)
            resumed = self.make_runner(root)
            resumed.training_config.early_stopping_min_delta = 0.1
            resumed.training_config.early_stopping_patience = 2
            resumed.step = 5
            resumed.evaluate.return_value['loss'] = 0.83
            resumed.validate_and_save_best()
            self.assertTrue(resumed.should_stop_early)

    def test_legacy_checkpoint_without_early_stopping_state(self):
        with project_temporary_directory() as folder:
            root = Path(folder)
            runner = self.make_runner(root)
            runner.min_loss = 0.5
            runner.save_checkpoint(0, 0)
            path = root / 'checkpoints/checkpoint-0.pt'
            state = torch.load(path, weights_only=True)
            del state['early_stopping_best_loss']
            del state['early_stopping_bad_evals']
            torch.save(state, path)
            resumed = self.make_runner(root)
            self.assertEqual(resumed.early_stopping_best_loss, 0.5)
            self.assertEqual(resumed.early_stopping_bad_evals, 0)

    def test_replace_retries_windows_sharing_lock(self):
        with project_temporary_directory() as folder:
            temporary = Path(folder) / 'checkpoint.pt.tmp'
            destination = Path(folder) / 'checkpoint.pt'
            temporary.write_bytes(b'new checkpoint')
            destination.write_bytes(b'previous checkpoint')
            original_replace = Path.replace
            locked = OSError('File is in use')
            locked.winerror = 32
            attempts = []

            def replace(source, target):
                attempts.append(source)
                if len(attempts) < 3:
                    self.assertEqual(destination.read_bytes(), b'previous checkpoint')
                    raise locked
                return original_replace(source, target)

            with patch.object(Path, 'replace', replace), patch('src.runner.train.time.sleep'):
                replace_checkpoint(temporary, destination)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(destination.read_bytes(), b'new checkpoint')
            self.assertFalse(temporary.exists())

    def test_replace_stops_after_persistent_lock(self):
        temporary = Mock()
        locked = OSError('File is in use')
        locked.winerror = 32
        temporary.replace.side_effect = locked
        with patch('src.runner.train.time.sleep'), self.assertRaises(OSError):
            replace_checkpoint(temporary, Path('checkpoint.pt'))
        self.assertEqual(temporary.replace.call_count, 8)
        temporary.unlink.assert_not_called()

    def test_replace_does_not_retry_other_errors(self):
        temporary = Mock()
        temporary.replace.side_effect = PermissionError('Access denied')
        with patch('src.runner.train.time.sleep') as sleep, self.assertRaises(PermissionError):
            replace_checkpoint(temporary, Path('checkpoint.pt'))
        temporary.replace.assert_called_once()
        sleep.assert_not_called()

    def make_runner(self, directory, device='cpu'):
        torch.manual_seed(123)
        model = TinyModel().to(device)
        data = [
            {'inputs': torch.tensor([i / 210, 1., -1.]), 'labels': i % 2}
            for i in range(210)
        ]
        loader = DataLoader(data, batch_size=2, shuffle=True, generator=torch.Generator())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        config = TrainConfig(epochs=1, eval_steps=100, output_dir=directory, log_dir=directory / 'logs')
        runner = Trainer(model, Mock(), optimizer, loader, loader, device, config)
        runner.evaluate = Mock(return_value={'loss': 0.5, 'metrics': {'accuracy': 0.5, 'f1': 0.5}})
        runner.save_model = Mock()
        return runner

    def assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_state_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_interrupted_training_matches_uninterrupted(self):
        with project_temporary_directory() as folder:
            root = Path(folder)
            baseline = self.make_runner(root / 'baseline')
            baseline.train()
            self.assertEqual(baseline.evaluate.call_count, 2)  # step 100 和轮末 105
            interrupted = self.make_runner(root / 'resume')
            original_save = interrupted.save_checkpoint

            def save_then_interrupt(*args):
                original_save(*args)
                raise InterruptedError('Simulated shutdown after step 100')

            interrupted.save_checkpoint = save_then_interrupt
            with self.assertRaises(InterruptedError):
                interrupted.train()
            self.assertTrue((root / 'resume/checkpoints/checkpoint-100.pt').exists())
            resumed = self.make_runner(root / 'resume')
            self.assertEqual((resumed.step, resumed.next_batch), (100, 100))
            self.assertEqual(resumed.last_eval_step, 100)
            resumed.train()
            self.assertEqual(resumed.evaluate.call_count, 1)
            self.assertEqual(resumed.step, 105)
            self.assert_state_equal(baseline.model.state_dict(), resumed.model.state_dict())
            self.assert_state_equal(baseline.optimizer.state_dict(), resumed.optimizer.state_dict())
            self.assertEqual(resumed.min_loss, 0.5)

            # 最新文件损坏时退回较早的完整检查点；临时文件不参与恢复。
            checkpoint_dir = root / 'resume/checkpoints'
            (checkpoint_dir / 'checkpoint-999.pt').write_bytes(b'broken')
            (checkpoint_dir / 'checkpoint-1000.pt.tmp').write_bytes(b'incomplete')
            completed = self.make_runner(root / 'resume')
            self.assertEqual((completed.step, completed.start_epoch), (105, 1))
            completed.train()
            completed.evaluate.assert_not_called()
            self.assertEqual(completed.step, 105)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_optimizer_and_rng_restore(self):
        with project_temporary_directory() as folder:
            root = Path(folder)
            runner = self.make_runner(root, 'cuda')
            batch = next(iter(runner.train_loader))
            batch = {k: v.to('cuda') for k, v in batch.items()}
            runner.model(**batch).loss.backward()
            runner.optimizer.step()
            runner.step = 100
            runner.save_checkpoint(0, 1, 1., 2)
            expected_rng = torch.cuda.get_rng_state()
            resumed = self.make_runner(root, 'cuda')
            self.assertTrue(torch.equal(expected_rng, torch.cuda.get_rng_state()))
            self.assert_state_equal(runner.optimizer.state_dict(), resumed.optimizer.state_dict())
            resumed.optimizer.zero_grad()
            resumed.model(**batch).loss.backward()
            resumed.optimizer.step()

    def test_epoch_boundary_does_not_repeat_evaluation(self):
        with project_temporary_directory() as folder:
            runner = self.make_runner(Path(folder))
            runner.training_config.eval_steps = 105
            runner.train()
            runner.evaluate.assert_called_once()
            self.assertEqual(runner.last_eval_step, 105)

    def test_retention_keeps_three_latest_and_best(self):
        with project_temporary_directory() as folder:
            root = Path(folder)
            runner = self.make_runner(root)
            (root / 'best').mkdir()
            (root / 'best/marker.txt').write_text('keep')
            for step in [100, 200, 300, 400]:
                runner.step = step
                runner.save_checkpoint(0, step)
            self.assertEqual(
                sorted(p.name for p in (root / 'checkpoints').glob('*.pt')),
                ['checkpoint-200.pt', 'checkpoint-300.pt', 'checkpoint-400.pt'],
            )
            self.assertTrue((root / 'best/marker.txt').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
